#!/usr/bin/env python3
# VERSION: conda-workers-2026-05-16-v5-sam3-single-workers-resume
# -*- coding: utf-8 -*-
"""
Case study script for ultrasound segmentation models.

功能：
1. 对每个 COCO test dataset 固定随机抽样 N=20 张图片；
2. 将抽中的原图和 subset COCO annotation 保存到 output_root/sampled_inputs；
3. 分别运行 UniBiomed、BiomedParse、SAM3、Medical-SAM3、US-SAM3；
4. 保存每个模型每张图的二值 mask 和 overlay 图；
5. 为每张样本生成一张横向拼接图：
   Original | Ground Truth | UniBiomed | BiomedParse | SAM3 | Medical-SAM3 | US-SAM3

说明：
- 本脚本尽量复用你已有三个 eval 脚本中的加载/推理逻辑。
- SAM3 相关模型通过 Hydra trainer.run_val() 在 sampled subset 上跑，随后读取 dumps/coco_predictions_segm.json。
- UniBiomed / BiomedParse 直接对 sampled images 做推理，并保存 mask/overlay。
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import csv
import io
import json
import random
import shutil
import sys
import subprocess
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


# ========================= 默认路径 =========================
DEFAULT_DATASETS_ROOT = "/home/Data2/zhuquanhao/datasets"
DEFAULT_OUTPUT_ROOT = "/home/Data2/zhuquanhao/case_study_outputs"

DEFAULT_UNIBIOMED_MODEL_PATH = "/home/external_hd/zhuquanhao/models/UniBiomed"
DEFAULT_BIOMEDPARSE_CODE_DIR = "/home/Data2/zhuquanhao/BiomedParse"
DEFAULT_BIOMEDPARSE_WEIGHT = "/home/Data2/zhuquanhao/BiomedParse/weights/biomedparse_v1.pt"
DEFAULT_BIOMEDPARSE_CONFIG = "configs/biomedparse_inference.yaml"

DEFAULT_SAM3_CODE_DIR = "/home/Data2/zhuquanhao/sam3/code"
DEFAULT_SAM3_CONFIG = "/home/Data2/zhuquanhao/sam3/configs/internal_datasets/config.yaml"
DEFAULT_SAM3_WEIGHTS = {
    "SAM3": "/home/Data2/zhuquanhao/sam3/weight/sam3.pt",
    "Medical-SAM3": "/home/Data2/zhuquanhao/Medical-SAM3/weight/Medical-SAM3.pt",
    "US-SAM3": "/home/Data2/zhuquanhao/sam3/logs/internal_datasets_exp_v2/checkpoints/checkpoint_2.pt",
}

# Mask post-processing defaults, aligned with your eval scripts.
MASK_THRESHOLD = 0.5
BIOMEDPARSE_BINARIZE_THRESHOLD = 0.5
BIOMEDPARSE_MIN_REGION_AREA = 10
BIOMEDPARSE_SELECTION_MODE = "top1"
TOP_K = 1
SCORE_THRESHOLD = 0.5

PANEL_NAMES = [
    "Original", "Ground Truth", "UniBiomed", "BiomedParse", "SAM3", "Medical-SAM3", "US-SAM3"
]


# ========================= dataset list =========================
def get_all_datasets(datasets_root: str):
    organs_datasets = {
        "Lung": ["LUSS_coco"],
        "Muscle": ["STMUS_NDA_coco", "LUMINOUS_coco", "FALLMUD_coco"],
        "Abdomen": ["AbdomenUS_coco"],
        "Ovarian": ["OTU_2d_coco", "OTU_3d_coco"],
        "Cardiac": ["EchoNet_Dynamic_coco", "CAMUS_coco", "Unity_coco", "EchoCP_coco", "EchoNet_Pediatric_coco", "CardiacUDC_coco"],
        "Gastrointestinal": ["GIST514_DB_coco", "c_trus_coco"],
        "Prostate": ["MicroSeg_coco", "RegPro_coco"],
        "Thyroid": ["Thyroid_US_Cineclip_coco", "KFGNet_coco", "TG3K_coco", "TN3K_coco", "Segthy_coco", "DDTI_coco"],
        "Liver": ["105US_tumor_coco", "liver_ultrasound_coco", "Annotated_Ultrasound_Liver_coco"],
        "Fetal": ["Fast_UNet_coco", "ACOUSLIC_coco", "fh_ps_coco", "FASS_coco", "focus_coco", "HC_coco"],
        "Nerve": ["US_Nerve_coco", "UPBD_coco"],
        "Breast": ["BUS_DatasetB_coco", "BUSI_coco", "BrEast_coco", "BUS_UCLM_coco", "BUS_BRA_coco", "STU_Hospital_coco", "BUS_UC_coco", "BUID_coco", "S1_coco"],
        "Carotid_artery": ["CCA_coco", "CCAUI_coco", "CUBS_coco"],
        "Kidney": ["Ultrasound_Normal_Kidney_coco", "KidneyUS_coco"],
    }

    test_datasets = {}
    for organ, datasets in organs_datasets.items():
        for dataset_name in datasets:
            key = f"{organ}/{dataset_name}"
            organ_path = organ
            if organ == "Fetal":
                organ_path = "fetal"
            if organ == "Liver":
                organ_path = "liver"
            if organ == "Muscle":
                organ_path = "muscle"
            test_datasets[key] = {
                "name": dataset_name,
                "img_folder": f"{datasets_root}/{organ_path}/Datasets/{dataset_name}/test/",
                "ann_file": f"{datasets_root}/{organ_path}/Datasets/{dataset_name}/test/_annotations.coco.json",
            }
    return test_datasets


def safe_key(dataset_key: str) -> str:
    return dataset_key.replace("/", "__")


# ========================= sampling and COCO subset =========================
def build_sampled_subset(dataset_key, dataset_info, out_root, samples_per_dataset=20, seed=2026, overwrite=False):
    """Fixed random sample for one dataset; create copied image folder and subset annotation."""
    dataset_dir = Path(out_root) / "sampled_inputs" / safe_key(dataset_key)
    img_out_dir = dataset_dir / "images"
    subset_ann_file = dataset_dir / "_annotations.sampled.coco.json"
    manifest_file = dataset_dir / "sample_manifest.json"

    if subset_ann_file.exists() and manifest_file.exists() and not overwrite:
        with open(manifest_file, "r") as f:
            manifest = json.load(f)
        return {
            **dataset_info,
            "img_folder": str(img_out_dir),
            "ann_file": str(subset_ann_file),
            "sampled_img_ids": manifest["sampled_img_ids"],
            "sample_manifest": str(manifest_file),
        }

    if not os.path.exists(dataset_info["ann_file"]):
        raise FileNotFoundError(dataset_info["ann_file"])

    coco = COCO(dataset_info["ann_file"])
    img_ids = sorted(coco.getImgIds())
    rng = random.Random(f"{seed}-{dataset_key}")
    sampled_img_ids = sorted(rng.sample(img_ids, min(samples_per_dataset, len(img_ids))))

    dataset_dir.mkdir(parents=True, exist_ok=True)
    img_out_dir.mkdir(parents=True, exist_ok=True)

    sampled_images = coco.loadImgs(sampled_img_ids)
    sampled_id_set = set(sampled_img_ids)
    sampled_anns = [ann for ann in coco.dataset.get("annotations", []) if ann["image_id"] in sampled_id_set]

    # Preserve original image IDs/category IDs, because SAM3 prediction JSON will refer to these IDs.
    subset = {
        "info": coco.dataset.get("info", {}),
        "licenses": coco.dataset.get("licenses", []),
        "images": sampled_images,
        "annotations": sampled_anns,
        "categories": coco.dataset.get("categories", []),
    }
    with open(subset_ann_file, "w") as f:
        json.dump(subset, f)

    copied = []
    for img in sampled_images:
        src = Path(dataset_info["img_folder"]) / img["file_name"]
        dst = img_out_dir / img["file_name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(img["file_name"])
        else:
            print(f"[WARN] Missing image: {src}")

    manifest = {
        "dataset_key": dataset_key,
        "seed": seed,
        "samples_per_dataset": samples_per_dataset,
        "sampled_img_ids": sampled_img_ids,
        "sampled_file_names": [img["file_name"] for img in sampled_images],
        "copied_file_names": copied,
        "source_img_folder": dataset_info["img_folder"],
        "source_ann_file": dataset_info["ann_file"],
        "subset_ann_file": str(subset_ann_file),
        "sampled_img_folder": str(img_out_dir),
    }
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    return {
        **dataset_info,
        "img_folder": str(img_out_dir),
        "ann_file": str(subset_ann_file),
        "sampled_img_ids": sampled_img_ids,
        "sample_manifest": str(manifest_file),
    }


def prepare_sampled_datasets(datasets, out_root, samples_per_dataset, seed, only_dataset=None, overwrite=False):
    if only_dataset:
        datasets = {k: v for k, v in datasets.items() if k == only_dataset}
        if not datasets:
            raise ValueError(f"Dataset key not found: {only_dataset}")

    sampled = {}
    for key, info in tqdm(datasets.items(), desc="Sampling datasets"):
        try:
            sampled[key] = build_sampled_subset(key, info, out_root, samples_per_dataset, seed, overwrite=overwrite)
        except Exception as e:
            print(f"[WARN] Skip {key}: {repr(e)}")
    return sampled


# ========================= mask utilities =========================
def gt_mask_for_image(coco: COCO, img_id: int, cat_id=None):
    img_info = coco.loadImgs(img_id)[0]
    h, w = int(img_info["height"]), int(img_info["width"])
    if cat_id is None:
        ann_ids = coco.getAnnIds(imgIds=img_id)
    else:
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[cat_id])
    gt = np.zeros((h, w), dtype=np.uint8)
    for ann in coco.loadAnns(ann_ids):
        gt = np.maximum(gt, coco.annToMask(ann).astype(np.uint8))
    return gt


def image_categories(coco: COCO):
    img_to_cats = defaultdict(set)
    for ann in coco.dataset.get("annotations", []):
        img_to_cats[int(ann["image_id"])].add(int(ann["category_id"]))
    cats = {cat["id"]: cat["name"] for cat in coco.loadCats(coco.getCatIds())}
    return img_to_cats, cats


def ensure_rgb_image(img_path, h=None, w=None):
    image = Image.open(img_path).convert("RGB")
    if h is not None and w is not None and image.size != (w, h):
        image = image.resize((w, h), Image.Resampling.BILINEAR)
    return image


def as_numpy_mask(mask):
    import torch
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().float().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            mask = np.max(mask, axis=0)
    return mask.astype(np.float32)


def binarize_and_resize_mask(mask, target_h, target_w, threshold=MASK_THRESHOLD):
    if mask is None:
        return np.zeros((target_h, target_w), dtype=np.uint8)
    mask = np.asarray(mask)
    if mask.dtype == np.bool_:
        binary = mask.astype(np.uint8)
    else:
        mask = mask.astype(np.float32)
        if mask.size == 0:
            return np.zeros((target_h, target_w), dtype=np.uint8)
        if mask.max() > 1.0:
            binary = (mask > 0).astype(np.uint8)
        else:
            binary = (mask > threshold).astype(np.uint8)
    if binary.shape != (target_h, target_w):
        binary = cv2.resize(binary, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (binary > 0).astype(np.uint8)


def extract_connected_components(pred_mask, binarize_thr=BIOMEDPARSE_BINARIZE_THRESHOLD, min_area=BIOMEDPARSE_MIN_REGION_AREA):
    from skimage.measure import label as connected_components
    pred_mask = as_numpy_mask(pred_mask)
    binary_mask = (pred_mask > binarize_thr).astype(np.uint8)
    labeled_mask = connected_components(binary_mask)
    candidates = []
    for region_id in range(1, int(labeled_mask.max()) + 1):
        region_mask = (labeled_mask == region_id).astype(np.uint8)
        area = int(region_mask.sum())
        if area < min_area:
            continue
        score = float(pred_mask[region_mask > 0].mean()) if area > 0 else 0.0
        candidates.append({"mask": region_mask, "score": score, "area": area})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def select_mask_candidates(candidates, mode=BIOMEDPARSE_SELECTION_MODE, top_k=TOP_K, score_thr=SCORE_THRESHOLD):
    if not candidates:
        return []
    mode = mode.lower()
    if mode == "all":
        return candidates
    if mode == "top1":
        return candidates[:1]
    if mode == "topk":
        return candidates[:top_k]
    if mode == "threshold":
        return [c for c in candidates if c["score"] >= score_thr]
    if mode == "topk_threshold":
        return [c for c in candidates if c["score"] >= score_thr][:top_k]
    raise ValueError(f"Unknown mode: {mode}")


def merge_candidates(candidates, h, w):
    merged = np.zeros((h, w), dtype=np.uint8)
    for c in candidates:
        m = c["mask"].astype(np.uint8)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        merged = np.maximum(merged, m)
    return merged


def decode_coco_pred_mask(ann, h, w):
    if "segmentation" not in ann:
        return None
    rle = ann["segmentation"]
    if isinstance(rle, dict) and "counts" in rle and isinstance(rle["counts"], list):
        rle = maskUtils.frPyObjects(rle, h, w)
    m = maskUtils.decode(rle)
    if m.ndim == 3:
        m = np.max(m, axis=2)
    if m.shape != (h, w):
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


def load_prediction_masks_from_coco(pred_file, coco_gt, mode="top1", top_k=1, score_thr=0.5):
    """Return {(img_id): union selected mask} from COCO result json."""
    masks = {}
    if not os.path.exists(pred_file):
        print(f"[WARN] Prediction file missing: {pred_file}")
        return masks
    with open(pred_file, "r") as f:
        preds = json.load(f)
    by_img = defaultdict(list)
    for ann in preds:
        by_img[int(ann["image_id"])].append(ann)
    for img_id in coco_gt.getImgIds():
        info = coco_gt.loadImgs(img_id)[0]
        h, w = int(info["height"]), int(info["width"])
        selected = sorted(by_img.get(int(img_id), []), key=lambda x: float(x.get("score", 0.0)), reverse=True)
        if mode == "top1":
            selected = selected[:1]
        elif mode == "topk":
            selected = selected[:top_k]
        elif mode == "threshold":
            selected = [a for a in selected if float(a.get("score", 0.0)) >= score_thr]
        elif mode == "topk_threshold":
            selected = [a for a in selected if float(a.get("score", 0.0)) >= score_thr][:top_k]
        elif mode == "all":
            pass
        else:
            raise ValueError(mode)
        merged = np.zeros((h, w), dtype=np.uint8)
        for ann in selected:
            m = decode_coco_pred_mask(ann, h, w)
            if m is not None:
                merged = np.maximum(merged, m)
        masks[int(img_id)] = merged
    return masks


# ========================= visualization =========================
def colorize_overlay(image_rgb, mask, alpha=0.45, color=(255, 0, 0)):
    arr = np.asarray(image_rgb).copy()
    if mask is None:
        return Image.fromarray(arr)
    mask = (mask > 0)
    overlay = arr.copy()
    overlay[mask] = (np.array(color) * alpha + overlay[mask] * (1 - alpha)).astype(np.uint8)
    return Image.fromarray(overlay)


def add_title(panel: Image.Image, title: str, title_h=34):
    panel = panel.convert("RGB")
    out = Image.new("RGB", (panel.width, panel.height + title_h), "white")
    out.paste(panel, (0, title_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    return out


def save_mask_and_overlay(image_pil, mask, model_name, dataset_key, img_info, out_root):
    model_dir = Path(out_root) / "model_outputs" / model_name / safe_key(dataset_key)
    mask_dir = model_dir / "masks"
    overlay_dir = model_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(img_info["file_name"]).stem
    mask_u8 = ((mask > 0).astype(np.uint8) * 255)
    Image.fromarray(mask_u8).save(mask_dir / f"{stem}.png")
    colorize_overlay(image_pil, mask).save(overlay_dir / f"{stem}.png")


def make_comparison_figure(dataset_key, img_info, image_pil, gt_mask, model_masks, out_root, panel_size=256):
    panels = []
    for name in PANEL_NAMES:
        if name == "Original":
            p = image_pil.copy()
        elif name == "Ground Truth":
            p = colorize_overlay(image_pil, gt_mask, color=(0, 255, 0))
        else:
            p = colorize_overlay(image_pil, model_masks.get(name), color=(255, 0, 0))
        p.thumbnail((panel_size, panel_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (panel_size, panel_size), "white")
        canvas.paste(p, ((panel_size - p.width) // 2, (panel_size - p.height) // 2))
        panels.append(add_title(canvas, name))

    gap = 8
    total_w = len(panels) * panel_size + (len(panels) - 1) * gap
    total_h = panels[0].height
    fig = Image.new("RGB", (total_w, total_h), "white")
    x = 0
    for p in panels:
        fig.paste(p, (x, 0))
        x += panel_size + gap

    compare_dir = Path(out_root) / "comparison_figures" / safe_key(dataset_key)
    compare_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(img_info["file_name"]).stem
    fig.save(compare_dir / f"{stem}.png")


# ========================= UniBiomed =========================
def load_unibiomed(model_path):
    import torch
    from transformers import AutoModel, AutoTokenizer
    print(f"[INFO] Loading UniBiomed: {model_path}")
    try:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).eval().cuda()
    except TypeError:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def unibiomed_extract_prediction_mask(pred_dict):
    masks = pred_dict.get("prediction_masks", None)
    if masks is None:
        return None
    try:
        mask = masks[0][0]
    except Exception:
        try:
            mask = masks[0]
        except Exception:
            return None
    return as_numpy_mask(mask)


def run_unibiomed_case_study(sampled_datasets, out_root, model_path, prompt_template, mask_threshold=0.5, debug=False):
    import torch
    model, tokenizer = load_unibiomed(model_path)
    all_masks = defaultdict(dict)

    for dataset_key, info in sampled_datasets.items():
        coco = COCO(info["ann_file"])
        img_to_cats, cats = image_categories(coco)
        for img_id in tqdm(coco.getImgIds(), desc=f"UniBiomed {dataset_key}"):
            img_info = coco.loadImgs(img_id)[0]
            h, w = int(img_info["height"]), int(img_info["width"])
            img_path = os.path.join(info["img_folder"], img_info["file_name"])
            if not os.path.exists(img_path):
                continue
            image = ensure_rgb_image(img_path, h, w)
            merged = np.zeros((h, w), dtype=np.uint8)
            for cat_id in sorted(img_to_cats.get(img_id, [])):
                cat_name = cats.get(cat_id, "object")
                text = prompt_template.format(category=cat_name)
                try:
                    with torch.no_grad():
                        pred_dict = model.predict_forward(image=image, text=text, tokenizer=tokenizer)
                    raw_mask = unibiomed_extract_prediction_mask(pred_dict)
                    pred_mask = binarize_and_resize_mask(raw_mask, h, w, threshold=mask_threshold)
                    merged = np.maximum(merged, pred_mask)
                except Exception as e:
                    if debug:
                        print(f"[WARN] UniBiomed failed {dataset_key} img={img_id} cat={cat_name}: {repr(e)}")
            all_masks[dataset_key][int(img_id)] = merged
            save_mask_and_overlay(image, merged, "UniBiomed", dataset_key, img_info, out_root)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return all_masks


# ========================= BiomedParse =========================
def load_biomedparse(weight_path, config_path):
    import torch
    from modeling.BaseModel import BaseModel
    from modeling import build_model
    from utilities.distributed import init_distributed
    from utilities.arguments import load_opt_from_config_files
    from utilities.constants import BIOMED_CLASSES

    print(f"[INFO] Loading BiomedParse: {weight_path}")
    opt = load_opt_from_config_files([config_path])
    opt = init_distributed(opt)
    model = BaseModel(opt, build_model(opt)).from_pretrained(weight_path).eval().cuda()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            BIOMED_CLASSES + ["background"], is_eval=True
        )
    return model


def run_biomedparse_case_study(sampled_datasets, out_root, weight_path, config_path, biomedparse_code_dir=DEFAULT_BIOMEDPARSE_CODE_DIR, debug=False):
    import torch
    biomedparse_code_dir = os.path.abspath(biomedparse_code_dir)
    if biomedparse_code_dir not in sys.path:
        sys.path.insert(0, biomedparse_code_dir)
    os.chdir(biomedparse_code_dir)
    if not os.path.isabs(config_path):
        config_path = os.path.join(biomedparse_code_dir, config_path)
    print(f"[INFO] BiomedParse cwd: {os.getcwd()}")
    print(f"[INFO] BiomedParse sys.path[0]: {sys.path[0]}")
    from inference_utils.inference import interactive_infer_image

    model = load_biomedparse(weight_path, config_path)
    all_masks = defaultdict(dict)

    for dataset_key, info in sampled_datasets.items():
        coco = COCO(info["ann_file"])
        img_to_cats, cats = image_categories(coco)
        for img_id in tqdm(coco.getImgIds(), desc=f"BiomedParse {dataset_key}"):
            img_info = coco.loadImgs(img_id)[0]
            h, w = int(img_info["height"]), int(img_info["width"])
            img_path = os.path.join(info["img_folder"], img_info["file_name"])
            if not os.path.exists(img_path):
                continue
            image = ensure_rgb_image(img_path, h, w)
            merged = np.zeros((h, w), dtype=np.uint8)
            for cat_id in sorted(img_to_cats.get(img_id, [])):
                cat_name = cats.get(cat_id, "object")
                try:
                    pred_masks = interactive_infer_image(model, image, [cat_name])
                    candidates = []
                    for pm in pred_masks:
                        candidates.extend(extract_connected_components(pm))
                    candidates.sort(key=lambda x: x["score"], reverse=True)
                    selected = select_mask_candidates(candidates)
                    merged = np.maximum(merged, merge_candidates(selected, h, w))
                except Exception as e:
                    if debug:
                        print(f"[WARN] BiomedParse failed {dataset_key} img={img_id} cat={cat_name}: {repr(e)}")
            all_masks[dataset_key][int(img_id)] = merged
            save_mask_and_overlay(image, merged, "BiomedParse", dataset_key, img_info, out_root)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return all_masks


# ========================= SAM3 family =========================
def init_sam3_trainer(model_name, checkpoint_path, config_path, output_dir, sam3_code_dir):
    if sam3_code_dir and sam3_code_dir not in sys.path:
        sys.path.insert(0, sam3_code_dir)
    import torch  # noqa: F401
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from sam3.train.utils.train_utils import register_omegaconf_resolvers

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12345")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    try:
        register_omegaconf_resolvers()
    except Exception:
        pass

    config_dir = os.path.dirname(os.path.abspath(config_path))
    config_name = os.path.basename(config_path)
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=config_dir, version_base="1.2")

    overrides = [
        "trainer.mode=val",
        f"trainer.model.checkpoint_path={checkpoint_path}",
        f"launcher.experiment_log_dir={output_dir}",
        f"trainer.checkpoint.save_dir={output_dir}/checkpoints",
        "trainer.skip_first_val=False",
        "trainer.skip_saving_ckpts=True",
        "trainer.meters.val.roboflow100.detection.iou_type=segm",
        "+trainer.meters.val.roboflow100.detection.postprocessor.iou_type=segm",
        "+trainer.meters.val.roboflow100.detection.postprocessor.convert_mask_to_rle=true",
        "+trainer.meters.val.roboflow100.detection.postprocessor.use_original_sizes_mask=true",
    ]

    cfg = compose(config_name=config_name, overrides=overrides)
    cfg.launcher.num_nodes = 1
    cfg.launcher.gpus_per_node = 1
    print(f"[INFO] Instantiating {model_name} from {checkpoint_path}")
    return instantiate(cfg.trainer, _recursive_=False)


def run_one_sam3_model(model_name, checkpoint_path, sampled_datasets, out_root, config_path, sam3_code_dir, debug=False):
    import torch
    from hydra.utils import instantiate

    model_out = Path(out_root) / "sam3_raw_outputs" / model_name
    model_out.mkdir(parents=True, exist_ok=True)
    trainer = init_sam3_trainer(model_name, checkpoint_path, config_path, str(model_out), sam3_code_dir)
    all_masks = defaultdict(dict)

    for dataset_key, info in sampled_datasets.items():
        print(f"[INFO] {model_name}: running {dataset_key}")
        current_output_dir = model_out / safe_key(dataset_key)
        current_output_dir.mkdir(parents=True, exist_ok=True)

        trainer.data_conf.val.dataset.img_folder = info["img_folder"]
        trainer.data_conf.val.dataset.ann_file = info["ann_file"]
        if "pred_file_evaluators" in trainer.meters_conf.val.roboflow100.detection:
            trainer.meters_conf.val.roboflow100.detection.pred_file_evaluators[0].gt_path = info["ann_file"]
        trainer.meters_conf.val.roboflow100.detection.dump_dir = f"{current_output_dir}/dumps"
        trainer.logging_conf.log_dir = str(current_output_dir)

        trainer._setup_dataloaders()
        trainer.meters = instantiate(trainer.meters_conf, _convert_="all")

        f = io.StringIO()
        with redirect_stdout(f):
            try:
                trainer.run_val()
            except Exception as e:
                print(f"[ERROR] {model_name} failed on {dataset_key}: {repr(e)}")
                if debug:
                    import traceback
                    traceback.print_exc()
        if debug:
            print(f.getvalue())

        pred_file = current_output_dir / "dumps" / "coco_predictions_segm.json"
        coco = COCO(info["ann_file"])
        masks = load_prediction_masks_from_coco(
            str(pred_file), coco, mode="top1", top_k=1, score_thr=SCORE_THRESHOLD
        )
        for img_id, mask in masks.items():
            img_info = coco.loadImgs(img_id)[0]
            image = ensure_rgb_image(os.path.join(info["img_folder"], img_info["file_name"]), img_info["height"], img_info["width"])
            all_masks[dataset_key][int(img_id)] = mask
            save_mask_and_overlay(image, mask, model_name, dataset_key, img_info, out_root)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del trainer
    # Important: SAM3 trainer initializes torch.distributed. If we instantiate
    # another SAM3-family trainer in the same Python process, the default
    # process group must be destroyed first; otherwise PyTorch raises
    # "trying to initialize the default process group twice".
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
            print(f"[INFO] Destroyed torch distributed process group after {model_name}.")
    except Exception as e:
        if debug:
            print(f"[WARN] Failed to destroy process group after {model_name}: {repr(e)}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_masks


def run_sam3_family_case_study(sampled_datasets, out_root, weights, config_path, sam3_code_dir, debug=False):
    all_model_masks = {}
    for model_name, ckpt in weights.items():
        if not ckpt or not os.path.exists(ckpt):
            print(f"[WARN] Skip {model_name}, checkpoint not found: {ckpt}")
            all_model_masks[model_name] = defaultdict(dict)
            continue
        all_model_masks[model_name] = run_one_sam3_model(
            model_name, ckpt, sampled_datasets, out_root, config_path, sam3_code_dir, debug=debug
        )
    return all_model_masks



def load_saved_mask(out_root, model_name, dataset_key, img_info, h, w):
    """Load a mask saved by a worker process. Returns zeros if not found."""
    stem = Path(img_info["file_name"]).stem
    mask_path = Path(out_root) / "model_outputs" / model_name / safe_key(dataset_key) / "masks" / f"{stem}.png"
    if not mask_path.exists():
        return np.zeros((h, w), dtype=np.uint8)
    mask = np.asarray(Image.open(mask_path).convert("L"))
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def expected_mask_count(sampled_datasets):
    """Number of sampled images expected for a full per-image model output."""
    total = 0
    for _, info in sampled_datasets.items():
        try:
            coco = COCO(info["ann_file"])
            total += len(coco.getImgIds())
        except Exception:
            pass
    return total


def existing_mask_count(out_root, model_name, sampled_datasets):
    """Count saved mask PNG files for a model over the currently sampled datasets."""
    total = 0
    for dataset_key in sampled_datasets.keys():
        mask_dir = Path(out_root) / "model_outputs" / model_name / safe_key(dataset_key) / "masks"
        if mask_dir.exists():
            total += len(list(mask_dir.glob("*.png")))
    return total


def model_outputs_complete(out_root, model_names, sampled_datasets):
    """Return True if all model_names have saved masks for all sampled images."""
    expected = expected_mask_count(sampled_datasets)
    if expected <= 0:
        return False
    for model_name in model_names:
        found = existing_mask_count(out_root, model_name, sampled_datasets)
        if found < expected:
            print(f"[INFO] Existing {model_name} masks: {found}/{expected} -> will run")
            return False
        print(f"[INFO] Existing {model_name} masks: {found}/{expected} -> skip")
    return True

# ========================= comparison output =========================
def save_gt_outputs_and_comparisons(sampled_datasets, out_root, model_masks_by_name, panel_size=256):
    rows = []
    for dataset_key, info in sampled_datasets.items():
        coco = COCO(info["ann_file"])
        for img_id in tqdm(coco.getImgIds(), desc=f"Figures {dataset_key}"):
            img_info = coco.loadImgs(img_id)[0]
            h, w = int(img_info["height"]), int(img_info["width"])
            img_path = os.path.join(info["img_folder"], img_info["file_name"])
            if not os.path.exists(img_path):
                continue
            image = ensure_rgb_image(img_path, h, w)
            gt = gt_mask_for_image(coco, img_id)
            save_mask_and_overlay(image, gt, "Ground_Truth", dataset_key, img_info, out_root)

            masks = {}
            for model_name in ["UniBiomed", "BiomedParse", "SAM3", "Medical-SAM3", "US-SAM3"]:
                mask = model_masks_by_name.get(model_name, {}).get(dataset_key, {}).get(int(img_id), None)
                if mask is None:
                    mask = load_saved_mask(out_root, model_name, dataset_key, img_info, h, w)
                masks[model_name] = mask
            make_comparison_figure(dataset_key, img_info, image, gt, masks, out_root, panel_size=panel_size)
            rows.append({
                "dataset_key": dataset_key,
                "image_id": int(img_id),
                "file_name": img_info["file_name"],
                "comparison": str(Path(out_root) / "comparison_figures" / safe_key(dataset_key) / f"{Path(img_info['file_name']).stem}.png"),
            })

    summary = Path(out_root) / "case_study_index.csv"
    with open(summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset_key", "image_id", "file_name", "comparison"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Wrote index: {summary}")


# ========================= CLI =========================
def parse_args():
    parser = argparse.ArgumentParser("Case study for UniBiomed / BiomedParse / SAM3 family")
    parser.add_argument("--datasets-root", default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samples-per-dataset", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--only-dataset", type=str, default=None, help="e.g. Cardiac/CAMUS_coco")
    parser.add_argument("--overwrite-samples", action="store_true")
    parser.add_argument("--panel-size", type=int, default=256)

    parser.add_argument("--skip-unibiomed", action="store_true")
    parser.add_argument("--skip-biomedparse", action="store_true")
    parser.add_argument("--skip-sam3", action="store_true")

    parser.add_argument("--unibiomed-model-path", default=DEFAULT_UNIBIOMED_MODEL_PATH)
    parser.add_argument("--unibiomed-mask-threshold", type=float, default=MASK_THRESHOLD)
    parser.add_argument("--prompt-template", default="<image>Please segment {category} in ultrasound.")

    parser.add_argument("--biomedparse-code-dir", default=DEFAULT_BIOMEDPARSE_CODE_DIR)
    parser.add_argument("--biomedparse-weight", default=DEFAULT_BIOMEDPARSE_WEIGHT)
    parser.add_argument("--biomedparse-config", default=DEFAULT_BIOMEDPARSE_CONFIG)

    parser.add_argument("--sam3-code-dir", default=DEFAULT_SAM3_CODE_DIR)
    parser.add_argument("--sam3-config", default=DEFAULT_SAM3_CONFIG)
    parser.add_argument("--sam3-ckpt", default=DEFAULT_SAM3_WEIGHTS["SAM3"])
    parser.add_argument("--medical-sam3-ckpt", default=DEFAULT_SAM3_WEIGHTS["Medical-SAM3"])
    parser.add_argument("--us-sam3-ckpt", default=DEFAULT_SAM3_WEIGHTS["US-SAM3"])

    parser.add_argument("--cuda-visible-devices", default=None, help="Optional, e.g. 0. Must be set before model import.")

    # Conda-worker mode: controller samples once, then launches this same script in separate conda envs.
    parser.add_argument("--use-conda-workers", action="store_true", help="Run each model family in its own conda environment via conda run.")
    parser.add_argument("--worker-stage", choices=["unibiomed", "biomedparse", "sam3", "sam3_model"], default=None, help="Internal worker stage. Usually set by the controller automatically.")
    parser.add_argument("--sam3-model-name", choices=["SAM3", "Medical-SAM3", "US-SAM3"], default=None, help="Internal: run only one SAM3-family model in a worker.")
    parser.add_argument("--run-output-root", default=None, help="Exact output directory for one run. Used internally to keep workers in the same folder.")
    parser.add_argument("--sam3-env", default="sam3")
    parser.add_argument("--biomedparse-env", default="biomedparse")
    parser.add_argument("--unibiomed-env", default="UniBiomed")
    parser.add_argument("--python-executable", default=sys.executable, help="Python executable used inside conda run; default is sys.executable.")
    parser.add_argument("--rerun-completed", action="store_true", help="By default, skip model workers whose masks already exist. Use this to force rerun.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _base_worker_cmd(args, out_root, worker_stage):
    """Arguments shared by conda worker subprocesses."""
    script_path = os.path.abspath(__file__)
    cmd = [
        "python", script_path,
        "--worker-stage", worker_stage,
        "--run-output-root", out_root,
        "--datasets-root", args.datasets_root,
        "--output-root", args.output_root,
        "--samples-per-dataset", str(args.samples_per_dataset),
        "--seed", str(args.seed),
        "--panel-size", str(args.panel_size),
        "--unibiomed-model-path", args.unibiomed_model_path,
        "--unibiomed-mask-threshold", str(args.unibiomed_mask_threshold),
        "--prompt-template", args.prompt_template,
        "--biomedparse-code-dir", args.biomedparse_code_dir,
        "--biomedparse-weight", args.biomedparse_weight,
        "--biomedparse-config", args.biomedparse_config,
        "--sam3-code-dir", args.sam3_code_dir,
        "--sam3-config", args.sam3_config,
        "--sam3-ckpt", args.sam3_ckpt,
        "--medical-sam3-ckpt", args.medical_sam3_ckpt,
        "--us-sam3-ckpt", args.us_sam3_ckpt,
    ]
    if getattr(args, "sam3_model_name", None):
        cmd += ["--sam3-model-name", args.sam3_model_name]
    if args.only_dataset:
        cmd += ["--only-dataset", args.only_dataset]
    if args.cuda_visible_devices is not None:
        cmd += ["--cuda-visible-devices", args.cuda_visible_devices]
    if args.debug:
        cmd += ["--debug"]
    return cmd


def run_conda_worker(env_name, args, out_root, worker_stage, sam3_model_name=None):
    old_sam3_model_name = getattr(args, "sam3_model_name", None)
    if sam3_model_name is not None:
        args.sam3_model_name = sam3_model_name
    try:
        cmd = ["conda", "run", "-n", env_name, "--no-capture-output"] + _base_worker_cmd(args, out_root, worker_stage)
    finally:
        args.sam3_model_name = old_sam3_model_name

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    cwd = None
    if worker_stage == "biomedparse":
        cwd = os.path.abspath(args.biomedparse_code_dir)
        env["PYTHONPATH"] = cwd + os.pathsep + env.get("PYTHONPATH", "")
    elif worker_stage in ("sam3", "sam3_model"):
        cwd = os.path.abspath(args.sam3_code_dir)
        env["PYTHONPATH"] = cwd + os.pathsep + env.get("PYTHONPATH", "")

    print("[INFO] Launching worker:", " ".join(cmd))
    print(f"[INFO] Worker cwd: {cwd or os.getcwd()}")
    if worker_stage in ("biomedparse", "sam3", "sam3_model"):
        print(f"[INFO] Worker PYTHONPATH prefix: {env.get('PYTHONPATH', '').split(os.pathsep)[0]}")

    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def main():
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if args.run_output_root:
        out_root = args.run_output_root
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = os.path.join(args.output_root, f"case_study_{timestamp}")
    os.makedirs(out_root, exist_ok=True)

    print(f"[INFO] Output root: {out_root}")
    datasets = get_all_datasets(args.datasets_root)
    sampled_datasets = prepare_sampled_datasets(
        datasets, out_root, args.samples_per_dataset, args.seed,
        only_dataset=args.only_dataset, overwrite=args.overwrite_samples,
    )
    print(f"[INFO] Prepared sampled datasets: {len(sampled_datasets)}")

    # Worker mode: run exactly one model family and exit. The controller will stitch figures later.
    if args.worker_stage == "unibiomed":
        run_unibiomed_case_study(
            sampled_datasets, out_root, args.unibiomed_model_path,
            args.prompt_template, mask_threshold=args.unibiomed_mask_threshold, debug=args.debug,
        )
        print("[DONE] UniBiomed worker finished.")
        return

    if args.worker_stage == "biomedparse":
        run_biomedparse_case_study(
            sampled_datasets, out_root, args.biomedparse_weight, args.biomedparse_config,
            biomedparse_code_dir=args.biomedparse_code_dir, debug=args.debug,
        )
        print("[DONE] BiomedParse worker finished.")
        return

    if args.worker_stage == "sam3_model":
        ckpt_map = {
            "SAM3": args.sam3_ckpt,
            "Medical-SAM3": args.medical_sam3_ckpt,
            "US-SAM3": args.us_sam3_ckpt,
        }
        if not args.sam3_model_name:
            raise ValueError("--sam3-model-name is required when --worker-stage sam3_model")
        ckpt = ckpt_map[args.sam3_model_name]
        run_one_sam3_model(
            args.sam3_model_name, ckpt, sampled_datasets, out_root, args.sam3_config, args.sam3_code_dir, debug=args.debug,
        )
        print(f"[DONE] {args.sam3_model_name} worker finished.")
        return

    if args.worker_stage == "sam3":
        # Backward-compatible family worker. The safer controller path below uses
        # one subprocess per SAM3-family checkpoint to avoid distributed-state reuse.
        sam3_weights = {
            "SAM3": args.sam3_ckpt,
            "Medical-SAM3": args.medical_sam3_ckpt,
            "US-SAM3": args.us_sam3_ckpt,
        }
        run_sam3_family_case_study(
            sampled_datasets, out_root, sam3_weights, args.sam3_config, args.sam3_code_dir, debug=args.debug,
        )
        print("[DONE] SAM3-family worker finished.")
        return

    # Controller mode with separate conda environments.
    if args.use_conda_workers:
        if not args.skip_unibiomed:
            if (not args.rerun_completed) and model_outputs_complete(out_root, ["UniBiomed"], sampled_datasets):
                print("[INFO] Skip UniBiomed worker because outputs already exist. Use --rerun-completed to force rerun.")
            else:
                run_conda_worker(args.unibiomed_env, args, out_root, "unibiomed")
        if not args.skip_biomedparse:
            if (not args.rerun_completed) and model_outputs_complete(out_root, ["BiomedParse"], sampled_datasets):
                print("[INFO] Skip BiomedParse worker because outputs already exist. Use --rerun-completed to force rerun.")
            else:
                run_conda_worker(args.biomedparse_env, args, out_root, "biomedparse")
        if not args.skip_sam3:
            # Run each SAM3-family checkpoint in a separate subprocess. This avoids
            # re-initializing torch.distributed inside the same Python interpreter
            # and also allows resume at per-model granularity.
            for sam3_model_name in ["SAM3", "Medical-SAM3", "US-SAM3"]:
                if (not args.rerun_completed) and model_outputs_complete(out_root, [sam3_model_name], sampled_datasets):
                    print(f"[INFO] Skip {sam3_model_name} worker because outputs already exist. Use --rerun-completed to force rerun.")
                else:
                    run_conda_worker(args.sam3_env, args, out_root, "sam3_model", sam3_model_name=sam3_model_name)

        # Load saved masks from disk and create final comparison figures in the current env.
        save_gt_outputs_and_comparisons(sampled_datasets, out_root, {}, panel_size=args.panel_size)
        print("[DONE] Case study finished with conda workers.")
        print(f"[DONE] Comparison figures: {os.path.join(out_root, 'comparison_figures')}")
        print(f"[DONE] Fixed sampled inputs: {os.path.join(out_root, 'sampled_inputs')}")
        print(f"[DONE] Model outputs: {os.path.join(out_root, 'model_outputs')}")
        return

    # Legacy single-process mode. This only works if one environment contains all dependencies.
    model_masks = {}

    if not args.skip_unibiomed:
        model_masks["UniBiomed"] = run_unibiomed_case_study(
            sampled_datasets, out_root, args.unibiomed_model_path,
            args.prompt_template, mask_threshold=args.unibiomed_mask_threshold, debug=args.debug,
        )
    else:
        model_masks["UniBiomed"] = defaultdict(dict)

    if not args.skip_biomedparse:
        model_masks["BiomedParse"] = run_biomedparse_case_study(
            sampled_datasets, out_root, args.biomedparse_weight, args.biomedparse_config,
            biomedparse_code_dir=args.biomedparse_code_dir, debug=args.debug,
        )
    else:
        model_masks["BiomedParse"] = defaultdict(dict)

    if not args.skip_sam3:
        sam3_weights = {
            "SAM3": args.sam3_ckpt,
            "Medical-SAM3": args.medical_sam3_ckpt,
            "US-SAM3": args.us_sam3_ckpt,
        }
        model_masks.update(run_sam3_family_case_study(
            sampled_datasets, out_root, sam3_weights, args.sam3_config, args.sam3_code_dir, debug=args.debug,
        ))
    else:
        model_masks["SAM3"] = defaultdict(dict)
        model_masks["Medical-SAM3"] = defaultdict(dict)
        model_masks["US-SAM3"] = defaultdict(dict)

    save_gt_outputs_and_comparisons(sampled_datasets, out_root, model_masks, panel_size=args.panel_size)
    print("[DONE] Case study finished.")
    print(f"[DONE] Comparison figures: {os.path.join(out_root, 'comparison_figures')}")
    print(f"[DONE] Fixed sampled inputs: {os.path.join(out_root, 'sampled_inputs')}")
    print(f"[DONE] Model outputs: {os.path.join(out_root, 'model_outputs')}")


if __name__ == "__main__":
    main()
