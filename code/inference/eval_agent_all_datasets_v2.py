import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import base64
import csv
import gc
import hashlib
import json
import re
from datetime import datetime
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils

from sam3.train.utils.train_utils import register_omegaconf_resolvers
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    Image as SAMImage,
    FindQueryLoaded,
    InferenceMetadata,
)

try:
    from sam3.agent.helpers.mask_overlap_removal import remove_overlapping_masks
except Exception:
    remove_overlapping_masks = None

METRIC_KEYS = ["mAP (0.50:0.95)", "AP (0.50)", "Mean IoU", "Mean Dice"]

# Prompt aliases only affect text prompts. COCO category ids remain unchanged.
CATEGORY_PROMPT_ALIASES = {
    "BB": "biceps brachii muscle",
    "BB_Healthy": "healthy biceps brachii muscle",
    "BB_Pathological": "pathological biceps brachii muscle",
    "GM": "gastrocnemius muscle",
    "GM_Healthy": "healthy gastrocnemius muscle",
    "GM_Pathological": "pathological gastrocnemius muscle",
    "TA": "tibialis anterior muscle",
    "TA_Healthy": "healthy tibialis anterior muscle",
    "TA_Pathological": "pathological tibialis anterior muscle",
    "CCA": "common carotid artery",
    "CCAUI": "common carotid artery intima-media region",
    "CUBS": "carotid ultrasound boundary structure",
    "TN3K": "thyroid nodule",
    "TG3K": "thyroid gland",
    "LV": "left ventricular cavity",
    "LA": "left atrium",
    "RV": "right ventricle",
    "RA": "right atrium",
    "MYO": "myocardium",
    "LVID": "left ventricular internal diameter",
    "IVS": "interventricular septum",
    "LVPW": "left ventricular posterior wall",
}


def setup_model(config_path, checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        register_omegaconf_resolvers()
    except Exception:
        pass

    config_dir = os.path.dirname(os.path.abspath(config_path))
    config_name = os.path.basename(config_path)
    initialize_config_dir(config_dir=config_dir, version_base="1.2")
    cfg = compose(config_name=config_name)
    cfg.trainer.model.checkpoint_path = None
    cfg.trainer.model.load_from_HF = False

    print("Loading SAM3 model...")
    model = instantiate(cfg.trainer.model)
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"Loaded checkpoint with missing={len(missing)}, unexpected={len(unexpected)}")
    model.to(device)
    model.eval()
    return model, device


def get_all_datasets(datasets_root="/home/Data2/zhuquanhao/datasets"):
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
        organ_path_name = organ
        if organ == "Fetal":
            organ_path_name = "fetal"
        elif organ == "Liver":
            organ_path_name = "liver"
        elif organ == "Muscle":
            organ_path_name = "muscle"
        for dataset_name in datasets:
            key = f"{organ}/{dataset_name}"
            test_datasets[key] = {
                "organ": organ,
                "name": dataset_name,
                "img_folder": f"{datasets_root}/{organ_path_name}/Datasets/{dataset_name}/test/",
                "ann_file": f"{datasets_root}/{organ_path_name}/Datasets/{dataset_name}/test/_annotations.coco.json",
            }
    return test_datasets




def build_organ_category_pools(test_datasets):
    """
    Build organ-level category pools across all datasets under the same organ.

    The API agent should not choose from every category in every dataset.  It first
    works inside the organ/domain of the current image, then chooses one target
    category from that organ-level pool.

    Each option has a stable string option_id.  If the option also exists in the
    current dataset, we later attach current_category_id so COCO evaluation can use
    the correct local category id.
    """
    pools = {}
    for dataset_key, info in test_datasets.items():
        organ = info.get("organ", dataset_key.split("/")[0])
        ann_file = info.get("ann_file")
        if not ann_file or not os.path.exists(ann_file):
            continue
        try:
            with open(ann_file, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        organ_pool = pools.setdefault(organ, {})
        for cat in data.get("categories", []):
            name = str(cat.get("name", "object")).strip() or "object"
            meaning = clean_category_prompt(name)
            key = meaning.lower()
            item = organ_pool.setdefault(key, {
                "organ": organ,
                "name": name,
                "meaning": meaning,
                "source_datasets": [],
                "source_names": [],
            })
            if dataset_key not in item["source_datasets"]:
                item["source_datasets"].append(dataset_key)
            if name not in item["source_names"]:
                item["source_names"].append(name)
    final = {}
    for organ, items in pools.items():
        options = []
        for i, (_, item) in enumerate(sorted(items.items(), key=lambda kv: kv[0]), start=1):
            item = dict(item)
            item["option_id"] = f"{organ}:{i}"
            options.append(item)
        final[organ] = options
    return final


def current_dataset_category_maps(coco_gt):
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    by_id = {int(c["id"]): c for c in cats}
    by_name = {str(c.get("name", "")).strip().lower(): c for c in cats}
    by_meaning = {clean_category_prompt(c.get("name", "object")).lower(): c for c in cats}
    return cats, by_id, by_name, by_meaning


def make_current_dataset_options(coco_gt):
    cats, _, _, _ = current_dataset_category_maps(coco_gt)
    out = []
    for c in cats:
        out.append({
            "option_id": str(int(c["id"])),
            "id": int(c["id"]),
            "current_category_id": int(c["id"]),
            "name": c.get("name", "object"),
            "meaning": clean_category_prompt(c.get("name", "object")),
            "source": "current_dataset",
        })
    return out


def make_organ_options_for_current_dataset(coco_gt, organ, organ_category_pools):
    """
    Return categories only from the current organ.  Options that are also present
    in the current dataset get current_category_id; other same-organ options are
    still shown to the agent for organ-level disambiguation, but cannot be used as
    COCO category ids unless matched back to the current dataset.
    """
    _, _, current_by_name, current_by_meaning = current_dataset_category_maps(coco_gt)
    raw_options = organ_category_pools.get(organ, [])
    out = []
    for opt in raw_options:
        name = opt.get("name", "object")
        meaning = opt.get("meaning", clean_category_prompt(name))
        current = current_by_meaning.get(meaning.lower()) or current_by_name.get(name.strip().lower())
        item = dict(opt)
        item["current_category_id"] = int(current["id"]) if current is not None else None
        item["current_category_name"] = current.get("name") if current is not None else None
        out.append(item)
    # Put current-dataset categories first to make valid COCO choices easier.
    out.sort(key=lambda x: (x.get("current_category_id") is None, x.get("meaning", "")))
    return out

def image_path_from_coco(img_folder, img_info):
    file_name = img_info.get("file_name", "")
    candidates = [
        os.path.join(img_folder, file_name),
        os.path.join(img_folder, os.path.basename(file_name)),
        file_name,
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return candidates[0]


def create_subset_coco_json(full_ann_file, subset_img_ids, out_file):
    with open(full_ann_file, "r") as f:
        data = json.load(f)
    subset = set(int(x) for x in subset_img_ids)
    new_data = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": [img for img in data.get("images", []) if int(img.get("id")) in subset],
        "annotations": [ann for ann in data.get("annotations", []) if int(ann.get("image_id")) in subset],
        "categories": data.get("categories", []),
    }
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(new_data, f)
    return out_file


def clean_category_prompt(category_name):
    if category_name in CATEGORY_PROMPT_ALIASES:
        return CATEGORY_PROMPT_ALIASES[category_name]
    s = str(category_name).strip()
    s = CATEGORY_PROMPT_ALIASES.get(s, s)
    s = s.replace("_", " ").replace("-", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s or "object"


def direct_simple_prompt_from_template(template, category_name):
    clean = clean_category_prompt(category_name)
    raw = str(category_name).replace("_", " ")
    return template.format(category=clean, raw_category=raw).strip()


def categories_for_image(coco_gt, img_id, mode):
    cats_all = coco_gt.loadCats(coco_gt.getCatIds())
    cat_by_id = {int(c["id"]): c for c in cats_all}
    if mode == "first":
        return [cats_all[0]] if cats_all else [{"id": 1, "name": "object"}]
    if mode == "all_dataset":
        return cats_all
    if mode == "image_gt":
        ann_ids = coco_gt.getAnnIds(imgIds=int(img_id))
        anns = coco_gt.loadAnns(ann_ids) if ann_ids else []
        ids = []
        for ann in anns:
            cid = int(ann.get("category_id", 1))
            if cid not in ids:
                ids.append(cid)
        return [cat_by_id[cid] for cid in ids if cid in cat_by_id]
    raise ValueError(f"Unknown category_mode: {mode}")


def candidate_categories(coco_gt, img_id, mode, dataset_info=None, organ_category_pools=None):
    """
    Category pool shown to the front agent.

    - image_gt/current_dataset/first are current-dataset-local choices.
    - organ_all restricts choices to the current organ/domain instead of mixing
      categories across all organs/datasets.
    """
    if mode == "image_gt":
        return [
            {
                "option_id": str(int(c["id"])),
                "id": int(c["id"]),
                "current_category_id": int(c["id"]),
                "name": c.get("name", "object"),
                "meaning": clean_category_prompt(c.get("name", "object")),
                "source": "image_gt",
            }
            for c in categories_for_image(coco_gt, img_id, "image_gt")
        ]
    if mode in ("all_dataset", "current_dataset"):
        return make_current_dataset_options(coco_gt)
    if mode == "first":
        return [
            {
                "option_id": str(int(c["id"])),
                "id": int(c["id"]),
                "current_category_id": int(c["id"]),
                "name": c.get("name", "object"),
                "meaning": clean_category_prompt(c.get("name", "object")),
                "source": "first",
            }
            for c in categories_for_image(coco_gt, img_id, "first")
        ]
    if mode == "organ_all":
        if dataset_info is None or organ_category_pools is None:
            raise ValueError("organ_all requires dataset_info and organ_category_pools")
        organ = dataset_info.get("organ", "")
        return make_organ_options_for_current_dataset(coco_gt, organ, organ_category_pools)
    raise ValueError(f"Unknown target_category_pool: {mode}")


def load_image_as_rgb(image_path):
    with Image.open(image_path) as im:
        im.load()
        return im.convert("RGB")


def build_batch_for_sam3(image_pil, text_prompt, device):
    orig_w, orig_h = image_pil.size
    image_resized = image_pil.resize((1008, 1008), resample=Image.BILINEAR)
    image_tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    image_normalized = (image_tensor - mean) / std

    find_query = FindQueryLoaded(
        query_text=text_prompt,
        image_id=0,
        object_ids_output=[],
        is_exhaustive=True,
        inference_metadata=InferenceMetadata(
            coco_image_id=0,
            original_image_id=0,
            original_category_id=0,
            original_size=(orig_h, orig_w),
            object_id=0,
            frame_index=0,
        ),
    )
    image_obj = SAMImage(data=image_normalized, objects=[], size=(1008, 1008))
    datapoint = Datapoint(find_queries=[find_query], images=[image_obj], raw_images=None)
    batch_input = collate_fn_api([datapoint], dict_key="all")["all"]
    if hasattr(batch_input, "img_batch"):
        batch_input.img_batch = batch_input.img_batch.to(device)
    for stage in batch_input.find_inputs:
        stage.input_boxes = stage.input_boxes.to(device)
        stage.input_boxes_mask = stage.input_boxes_mask.to(device)
        stage.input_boxes_label = stage.input_boxes_label.to(device)
        stage.input_points = stage.input_points.to(device)
        stage.input_points_mask = stage.input_points_mask.to(device)
        stage.img_ids = stage.img_ids.to(device)
    return batch_input, orig_h, orig_w


def extract_logits_masks(outputs, device):
    output = outputs[0]
    if "find_stages" in output:
        last_stage = output["find_stages"][-1]
        if isinstance(last_stage, list):
            pred_logits = last_stage[0]["pred_logits"]
            pred_masks = last_stage[0]["pred_masks"]
        else:
            pred_logits = last_stage["pred_logits"][0]
            pred_masks = last_stage["pred_masks"][0]
    elif "pred_logits" in output:
        pred_logits = output["pred_logits"]
        pred_masks = output["pred_masks"]
    else:
        pred_logits = torch.empty((0, 1), device=device)
        pred_masks = torch.empty((0, 1008, 1008), device=device)
    return pred_logits, pred_masks


def masks_to_serialized(pred_logits, pred_masks, orig_h, orig_w):
    if pred_logits.numel() > 0:
        if pred_logits.shape[-1] == 1:
            scores = torch.sigmoid(pred_logits).view(-1)
        else:
            scores = torch.softmax(pred_logits, dim=-1)[..., 0].view(-1)
    else:
        scores = torch.empty((0,), device=pred_masks.device if hasattr(pred_masks, "device") else "cpu")

    if pred_masks.numel() > 0:
        if pred_masks.ndim == 3:
            pred_masks = pred_masks.unsqueeze(1)
        elif pred_masks.ndim > 4:
            pred_masks = pred_masks.view(-1, 1, pred_masks.shape[-2], pred_masks.shape[-1])
        pred_masks = torch.nn.functional.interpolate(
            pred_masks.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False
        ).squeeze(1)
        if pred_masks.ndim > 3:
            pred_masks = pred_masks.view(-1, orig_h, orig_w)
        elif pred_masks.ndim == 2:
            pred_masks = pred_masks.unsqueeze(0)
        pred_masks_np = (pred_masks > 0).detach().cpu().numpy().astype(np.uint8)
    else:
        pred_masks_np = np.empty((0, orig_h, orig_w), dtype=np.uint8)

    scores_np = scores.detach().cpu().numpy().astype(float).tolist()
    boxes, rles, areas = [], [], []
    for m in pred_masks_np:
        ys, xs = np.where(m > 0)
        area = int(m.sum())
        areas.append(area)
        if len(xs) > 0:
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            boxes.append([x0, y0, x1 - x0 + 1, y1 - y0 + 1])
        else:
            boxes.append([0, 0, 0, 0])
        rle = maskUtils.encode(np.asfortranarray(m))
        rle["counts"] = rle["counts"].decode("utf-8")
        rles.append(rle)
    return {
        "orig_img_h": int(orig_h),
        "orig_img_w": int(orig_w),
        "pred_boxes": boxes,
        "pred_masks": rles,
        "pred_scores": scores_np,
        "pred_areas": areas,
    }


def sam3_predict_image(model, device, image_path, text_prompt, args):
    image_pil = load_image_as_rgb(image_path)
    batch_input, orig_h, orig_w = build_batch_for_sam3(image_pil, text_prompt, device)
    try:
        with torch.inference_mode():
            outputs = model(batch_input)
        pred_logits, pred_masks = extract_logits_masks(outputs, device)
        serialized = masks_to_serialized(pred_logits, pred_masks, orig_h, orig_w)
        if args.remove_overlap and remove_overlapping_masks is not None:
            tmp = {
                "orig_img_h": serialized["orig_img_h"],
                "orig_img_w": serialized["orig_img_w"],
                "pred_boxes": serialized["pred_boxes"],
                "pred_masks": [r["counts"] for r in serialized["pred_masks"]],
                "pred_scores": serialized["pred_scores"],
            }
            tmp = remove_overlapping_masks(tmp)
            masks, areas = [], []
            for counts in tmp.get("pred_masks", []):
                rle = {"size": [orig_h, orig_w], "counts": counts}
                masks.append(rle)
                try:
                    areas.append(int(maskUtils.area(rle)))
                except Exception:
                    areas.append(0)
            serialized["pred_boxes"] = tmp.get("pred_boxes", [])
            serialized["pred_masks"] = masks
            serialized["pred_scores"] = tmp.get("pred_scores", [])
            serialized["pred_areas"] = areas
        return serialized
    finally:
        for var in ["outputs", "pred_logits", "pred_masks", "batch_input", "image_pil"]:
            if var in locals():
                del locals()[var]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def collect_candidates(serialized, category_id, category_name, prompt, args):
    candidates = []
    h, w = serialized["orig_img_h"], serialized["orig_img_w"]
    max_area = int(h * w * args.max_area_ratio) if args.max_area_ratio and args.max_area_ratio > 0 else None
    for mask_idx, rle in enumerate(serialized.get("pred_masks", [])):
        area = int(serialized.get("pred_areas", [0] * len(serialized.get("pred_masks", [])))[mask_idx])
        if area < args.min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        score = float(serialized.get("pred_scores", [0.0] * len(serialized.get("pred_masks", [])))[mask_idx])
        box = serialized.get("pred_boxes", [[0, 0, 0, 0]] * len(serialized.get("pred_masks", [])))[mask_idx]
        candidates.append({
            "prompt": prompt,
            "mask_idx": mask_idx,
            "score": score,
            "area": area,
            "box": box,
            "rle": rle,
            "h": h,
            "w": w,
            "category_id": int(category_id),
            "category_name": category_name,
        })
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def select_top1(candidates):
    return candidates[:1] if candidates else []


def selected_to_coco(selected_items, image_id):
    preds = []
    for item in selected_items:
        h, w = int(item["h"]), int(item["w"])
        rle = item["rle"]
        if isinstance(rle, str):
            rle = {"size": [h, w], "counts": rle}
        else:
            rle = dict(rle)
            rle.setdefault("size", [h, w])
            if isinstance(rle.get("counts"), bytes):
                rle["counts"] = rle["counts"].decode("utf-8")
        preds.append({
            "image_id": int(image_id),
            "category_id": int(item["category_id"]),
            "segmentation": rle,
            "bbox": [float(x) for x in item.get("box", [0, 0, 0, 0])[:4]],
            "score": float(item.get("score", 1.0)),
        })
    return preds


def pil_to_data_url(img, fmt="JPEG", quality=90):
    buf = BytesIO()
    img = img.convert("RGB")
    img.save(buf, format=fmt, quality=quality, optimize=True)
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def image_path_to_data_url(image_path, max_side=768):
    with Image.open(image_path) as im:
        im.load()
        im = im.convert("RGB")
        if max_side and max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.BILINEAR)
        return pil_to_data_url(im, fmt="JPEG", quality=90)


def call_chat_api(messages, args, max_tokens=None):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"}
    payload = {
        "model": args.api_model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens or args.api_max_tokens,
    }
    resp = requests.post(args.api_url, headers=headers, json=payload, timeout=args.api_timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def extract_json_object(text):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def fallback_complex_instruction(clean_category, organ):
    return f"In this {organ.lower()} ultrasound image, identify and segment the anatomical structure corresponding to {clean_category}."


def cache_key(prefix, dataset_key, img_id, category_id, category_name, extra=None):
    raw = {
        "prefix": prefix,
        "dataset": dataset_key,
        "img_id": int(img_id),
        "category_id": int(category_id) if str(category_id).isdigit() else str(category_id),
        "category_name": str(category_name),
        "extra": extra or {},
    }
    return hashlib.md5(json.dumps(raw, sort_keys=True).encode("utf-8")).hexdigest()


def load_cache(cache_file):
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache_file, cache):
    if not cache_file:
        return
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def generate_complex_instruction(dataset_key, dataset_info, img_info, category_id, category_name, image_path, args, cache):
    clean = clean_category_prompt(category_name)
    organ = dataset_info.get("organ", dataset_key.split("/")[0])
    key = cache_key("complex_instruction", dataset_key, img_info.get("id"), category_id, category_name, {
        "include_image": bool(args.complex_include_image),
        "style": args.complex_instruction_style,
    })
    if key in cache:
        item = cache[key]
        return item.get("instruction", fallback_complex_instruction(clean, organ)), item.get("raw_response", ""), "cache"

    system_text = (
        "You generate synthetic user instructions for evaluating an ultrasound segmentation agent. "
        "Given the ground-truth target category, write ONE complex but unambiguous natural-language instruction that refers to the target. "
        "The instruction should be something a user might ask, not just the category name. "
        "Do not mention the exact raw category string unless it is already natural. "
        "Return ONLY valid JSON: {\"instruction\": \"...\", \"target_meaning\": \"...\"}."
    )
    user_text = (
        f"Dataset: {dataset_key}\n"
        f"Organ group: {organ}\n"
        f"Dataset name: {dataset_info.get('name', '')}\n"
        f"Image file: {img_info.get('file_name', '')}\n"
        f"Raw GT category: {category_name!r}\n"
        f"Clean target meaning: {clean!r}\n"
        f"Instruction style: {args.complex_instruction_style}\n\n"
        "Write one instruction under 30 words. It may use anatomical context, modality context, or relational phrasing, "
        "but it must still refer to only this one target."
    )
    messages = [{"role": "system", "content": system_text}]
    if args.complex_include_image:
        messages.append({"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_path_to_data_url(image_path, args.api_image_max_side)}},
        ]})
    else:
        messages.append({"role": "user", "content": user_text})

    try:
        raw = call_chat_api(messages, args, max_tokens=args.api_max_tokens)
        obj = extract_json_object(raw)
        instr = ""
        if isinstance(obj, dict):
            instr = str(obj.get("instruction", "")).strip()
        if not instr:
            instr = fallback_complex_instruction(clean, organ)
            status = "fallback_empty"
        else:
            status = "api"
    except Exception as e:
        raw = f"ERROR: {e}"
        instr = fallback_complex_instruction(clean, organ)
        status = "fallback_error"
        if not args.api_fallback_on_error:
            raise

    instr = re.sub(r"\s+", " ", instr).strip()[:args.max_instruction_chars]
    cache[key] = {"instruction": instr, "raw_response": raw, "status": status}
    return instr, raw, status


def parse_agent_output(text, target_cats):
    obj = extract_json_object(text)
    if not isinstance(obj, dict):
        return None

    chosen_option = obj.get("chosen_option_id", obj.get("option_id", None))
    cid = obj.get("chosen_category_id", obj.get("category_id", None))
    cname = obj.get("chosen_category_name", obj.get("category_name", ""))
    cmeaning = obj.get("chosen_category_meaning", obj.get("category_meaning", ""))
    prompt = obj.get("sam3_prompt", obj.get("prompt", obj.get("simple_prompt", "")))
    reason = obj.get("reason", "")
    chosen_organ = obj.get("chosen_organ", obj.get("organ", ""))

    by_option = {str(c.get("option_id", c.get("id", ""))): c for c in target_cats}
    by_current_id = {
        int(c["current_category_id"]): c
        for c in target_cats
        if c.get("current_category_id") is not None
    }
    by_name = {str(c.get("name", "")).strip().lower(): c for c in target_cats}
    by_current_name = {
        str(c.get("current_category_name", "")).strip().lower(): c
        for c in target_cats
        if c.get("current_category_name")
    }
    by_meaning = {str(c.get("meaning", "")).strip().lower(): c for c in target_cats}

    chosen_cat = None
    if chosen_option is not None:
        chosen_cat = by_option.get(str(chosen_option).strip())
    if chosen_cat is None:
        try:
            cid_int = int(cid)
            chosen_cat = by_current_id.get(cid_int)
        except Exception:
            pass
    if chosen_cat is None and cname:
        key = str(cname).strip().lower()
        chosen_cat = by_name.get(key) or by_current_name.get(key)
    if chosen_cat is None and cmeaning:
        chosen_cat = by_meaning.get(str(cmeaning).strip().lower())
    if chosen_cat is None and cname:
        key = clean_category_prompt(cname).lower()
        chosen_cat = by_meaning.get(key)
    if chosen_cat is None:
        return None

    current_category_id = chosen_cat.get("current_category_id")
    current_category_name = chosen_cat.get("current_category_name") or chosen_cat.get("name", "object")

    # If the organ-level option does not exist in the current dataset, the agent
    # made an out-of-dataset choice.  The caller will fallback for category_id,
    # but we still keep the generated prompt for debugging.
    if not isinstance(prompt, str) or not prompt.strip():
        prompt = chosen_cat.get("meaning") or clean_category_prompt(current_category_name)

    return {
        "option_id": str(chosen_cat.get("option_id", chosen_cat.get("id", ""))),
        "category_id": int(current_category_id) if current_category_id is not None else None,
        "category_name": current_category_name,
        "organ_pool_name": chosen_cat.get("name", current_category_name),
        "organ_pool_meaning": chosen_cat.get("meaning", clean_category_prompt(current_category_name)),
        "sam3_prompt": re.sub(r"\s+", " ", prompt).strip(),
        "reason": str(reason),
        "chosen_organ": str(chosen_organ),
    }


def agent_parse_instruction(dataset_key, dataset_info, img_info, image_path, complex_instruction, target_cats, args, cache):
    target_payload = []
    for c in target_cats:
        target_payload.append({
            "option_id": str(c.get("option_id", c.get("id", ""))),
            "current_category_id": c.get("current_category_id", c.get("id", None)),
            "name": c.get("name", "object"),
            "meaning": c.get("meaning", clean_category_prompt(c.get("name", "object"))),
            "current_dataset_name": c.get("current_category_name", c.get("name", "object")),
            "source_datasets": c.get("source_datasets", []),
        })
    extra = {"instruction": complex_instruction, "pool": args.target_category_pool, "include_image": True}
    key = cache_key("agent_parse", dataset_key, img_info.get("id"), 0, "instruction", extra)
    if key in cache:
        item = cache[key]
        return item.get("parsed"), item.get("raw_response", ""), "cache"


    system_text = (
        "You are a lightweight front-end agent for SAM3 ultrasound segmentation. "
        "You receive an image, a complex user instruction, the known organ/domain for this image, "
        "and a fixed list of valid target categories restricted to that organ/domain. "
        "First identify the organ/domain, then choose exactly one target category from the provided organ-level list, "
        "and rewrite the instruction into one short SAM3 text prompt. "
        "Do not segment masks. Do not invent categories outside the list. "
        "Return ONLY valid JSON with keys: chosen_organ, chosen_option_id, chosen_category_name, sam3_prompt, reason."
    )
    user_text = (
        f"Dataset: {dataset_key}\n"
        f"Organ group: {dataset_info.get('organ', '')}\n"
        f"Image file: {img_info.get('file_name', '')}\n"
        f"Complex user instruction: {complex_instruction!r}\n\n"
        "Valid target categories, as JSON array:\n"
        f"{json.dumps(target_payload, ensure_ascii=False)}\n\n"
        "Select the single best matching category id/name. Then write a concise SAM3 prompt, preferably a noun phrase or a short instruction. "
        "Keep sam3_prompt under 14 words."
    )
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_path_to_data_url(image_path, args.api_image_max_side)}},
        ]},
    ]

    try:
        raw = call_chat_api(messages, args, max_tokens=args.api_max_tokens)
        parsed = parse_agent_output(raw, target_cats)
        if parsed is None:
            status = "fallback_invalid"
        else:
            status = "api"
    except Exception as e:
        raw = f"ERROR: {e}"
        parsed = None
        status = "fallback_error"
        if not args.api_fallback_on_error:
            raise

    cache[key] = {"parsed": parsed, "raw_response": raw, "status": status}
    return parsed, raw, status


def decode_pred_mask(ann, h, w):
    rle = ann.get("segmentation")
    if rle is None:
        return None
    if isinstance(rle, dict) and isinstance(rle.get("counts"), list):
        rle = maskUtils.frPyObjects(rle, h, w)
    m = maskUtils.decode(rle)
    if m.ndim == 3:
        m = np.max(m, axis=2)
    if m.shape[:2] != (h, w):
        import cv2
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


def compute_iou_dice_from_coco(gt_file, pred_file):
    coco_gt = COCO(gt_file)
    with open(pred_file, "r") as f:
        preds = json.load(f)
    if not preds:
        return 0.0, 0.0
    coco_dt = coco_gt.loadRes(preds)
    ious, dices = [], []
    for img_id in coco_gt.getImgIds():
        img_info = coco_gt.loadImgs(img_id)[0]
        h, w = int(img_info["height"]), int(img_info["width"])
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        for ann in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id)):
            gt_mask = np.maximum(gt_mask, coco_gt.annToMask(ann).astype(np.uint8))
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        for ann in coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=img_id)):
            m = decode_pred_mask(ann, h, w)
            if m is not None:
                pred_mask = np.maximum(pred_mask, m)
        inter = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        pred_sum, gt_sum = pred_mask.sum(), gt_mask.sum()
        if union == 0:
            iou, dice = 1.0, 1.0
        else:
            iou = float(inter / union)
            dice = 0.0 if (pred_sum + gt_sum) == 0 else float(2 * inter / (pred_sum + gt_sum))
        ious.append(iou)
        dices.append(dice)
    return float(np.mean(ious)) if ious else 0.0, float(np.mean(dices)) if dices else 0.0


def compute_ap_from_coco(gt_file, pred_file):
    coco_gt = COCO(gt_file)
    with open(pred_file, "r") as f:
        preds = json.load(f)
    if not preds:
        return {"mAP (0.50:0.95)": 0.0, "AP (0.50)": 0.0}
    coco_dt = coco_gt.loadRes(preds)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="segm")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {"mAP (0.50:0.95)": float(evaluator.stats[0]), "AP (0.50)": float(evaluator.stats[1])}


def evaluate_prediction_file(gt_file, pred_file):
    metrics = compute_ap_from_coco(gt_file, pred_file)
    mean_iou, mean_dice = compute_iou_dice_from_coco(gt_file, pred_file)
    metrics["Mean IoU"] = mean_iou
    metrics["Mean Dice"] = mean_dice
    return metrics


def run_one_prompt(model, device, image_path, prompt, category_id, category_name, args):
    ser = sam3_predict_image(model, device, image_path, prompt, args)
    candidates = collect_candidates(ser, category_id, category_name, prompt, args)
    selected = select_top1(candidates)
    return selected, candidates


def run_dataset(dataset_key, dataset_info, model, device, args):
    print(f"\n--- Testing complex-instruction front agent on {dataset_key} ---")
    ann_file = dataset_info["ann_file"]
    img_folder = dataset_info["img_folder"]
    if not os.path.exists(ann_file):
        print(f"Warning: annotation file not found: {ann_file}")
        return None

    out_dir = os.path.join(args.output_dir, dataset_key.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)
    cache_file = os.path.join(out_dir, "complex_agent_cache.json")
    api_cache = load_cache(cache_file)

    coco_full = COCO(ann_file)
    img_ids = coco_full.getImgIds()
    if args.max_images and args.max_images > 0:
        img_ids = img_ids[:args.max_images]
        if args.subset_gt_for_max_images:
            eval_gt_file = create_subset_coco_json(ann_file, img_ids, os.path.join(out_dir, "subset_gt.json"))
            print(f"Using subset GT for max_images={args.max_images}: {eval_gt_file}")
        else:
            eval_gt_file = ann_file
            print("WARNING: max_images is set but full GT is used. Metrics will be artificially low.")
    else:
        eval_gt_file = ann_file

    direct_pred_file = os.path.join(out_dir, "coco_predictions_segm_A_complex_direct_top1.json")
    agent_pred_file = os.path.join(out_dir, "coco_predictions_segm_B_agent_parsed_top1.json")
    debug_file = os.path.join(out_dir, "per_image_debug.csv")

    print(f"GT category source for generating instructions: {args.gt_category_mode}")
    print(f"Target category pool shown to agent: {args.target_category_pool}")
    print("A: complex instruction directly into SAM3 top1")
    print("B: API agent sees image + complex instruction + target category list, then outputs simple SAM3 prompt; SAM3 top1")

    direct_predictions = []
    agent_predictions = []
    debug_fields = [
        "image_id", "file_name", "status", "gt_category_id", "gt_category_name",
        "complex_instruction", "complex_status",
        "direct_prompt", "direct_raw_masks", "direct_top1_score", "direct_top1_area",
        "agent_status", "agent_chosen_option_id", "agent_chosen_category_id", "agent_chosen_category_name", "agent_prompt",
        "agent_raw_masks", "agent_top1_score", "agent_top1_area",
        "agent_correct_category", "agent_choice_in_current_dataset", "target_pool_size", "error", "complex_raw_short", "agent_raw_short",
    ]

    with open(debug_file, "w", newline="") as fdebug:
        writer = csv.DictWriter(fdebug, fieldnames=debug_fields)
        writer.writeheader()
        for idx, img_id in enumerate(img_ids, start=1):
            img_info = coco_full.loadImgs(img_id)[0]
            image_path = image_path_from_coco(img_folder, img_info)
            file_name = img_info.get("file_name", os.path.basename(image_path))
            print(f"[{idx}/{len(img_ids)}] {os.path.basename(image_path)}")

            if not os.path.exists(image_path):
                writer.writerow({"image_id": img_id, "file_name": file_name, "status": "missing", "error": f"missing image: {image_path}"})
                continue

            gt_cats = categories_for_image(coco_full, img_id, args.gt_category_mode)
            if not gt_cats:
                writer.writerow({"image_id": img_id, "file_name": file_name, "status": "no_gt_category", "error": "no GT category for this image"})
                continue
            target_cats = candidate_categories(
                coco_full,
                img_id,
                args.target_category_pool,
                dataset_info=dataset_info,
                organ_category_pools=getattr(args, "organ_category_pools", None),
            )

            for gt_cat in gt_cats:
                gt_category_id = int(gt_cat["id"])
                gt_category_name = gt_cat.get("name", "object")
                try:
                    complex_instr, complex_raw, complex_status = generate_complex_instruction(
                        dataset_key, dataset_info, img_info, gt_category_id, gt_category_name, image_path, args, api_cache
                    )

                    # A. Complex instruction is fed directly to SAM3. Category id is the known GT category,
                    # because this branch tests whether SAM3 can understand the complex user instruction itself.
                    direct_selected, direct_candidates = run_one_prompt(
                        model, device, image_path, complex_instr, gt_category_id, gt_category_name, args
                    )
                    direct_predictions.extend(selected_to_coco(direct_selected, img_id))

                    # B. Agent parses the instruction into one category and a simple SAM3 prompt.
                    parsed, agent_raw, agent_status = agent_parse_instruction(
                        dataset_key, dataset_info, img_info, image_path, complex_instr, target_cats, args, api_cache
                    )
                    if parsed is None:
                        # Fallback keeps pipeline running but makes the fallback visible in debug.
                        parsed = {
                            "option_id": "fallback_gt",
                            "category_id": gt_category_id,
                            "category_name": gt_category_name,
                            "sam3_prompt": direct_simple_prompt_from_template(args.fallback_simple_prompt_template, gt_category_name),
                            "reason": "fallback to GT category after invalid API output",
                        }
                    choice_in_current_dataset = parsed.get("category_id") is not None
                    eval_category_id = int(parsed["category_id"]) if choice_in_current_dataset else gt_category_id
                    eval_category_name = parsed.get("category_name") or gt_category_name
                    agent_selected, agent_candidates = run_one_prompt(
                        model, device, image_path, parsed["sam3_prompt"], eval_category_id, eval_category_name, args
                    )
                    agent_predictions.extend(selected_to_coco(agent_selected, img_id))

                    dtop = direct_selected[0] if direct_selected else None
                    atop = agent_selected[0] if agent_selected else None
                    writer.writerow({
                        "image_id": img_id,
                        "file_name": file_name,
                        "status": "ok",
                        "gt_category_id": gt_category_id,
                        "gt_category_name": gt_category_name,
                        "complex_instruction": complex_instr,
                        "complex_status": complex_status,
                        "direct_prompt": complex_instr,
                        "direct_raw_masks": len(direct_candidates),
                        "direct_top1_score": f"{dtop['score']:.6f}" if dtop else "",
                        "direct_top1_area": int(dtop["area"]) if dtop else "",
                        "agent_status": agent_status,
                        "agent_chosen_option_id": parsed.get("option_id", ""),
                        "agent_chosen_category_id": parsed.get("category_id", ""),
                        "agent_chosen_category_name": parsed.get("category_name", ""),
                        "agent_prompt": parsed["sam3_prompt"],
                        "agent_raw_masks": len(agent_candidates),
                        "agent_top1_score": f"{atop['score']:.6f}" if atop else "",
                        "agent_top1_area": int(atop["area"]) if atop else "",
                        "agent_correct_category": int(parsed.get("category_id") == gt_category_id),
                        "agent_choice_in_current_dataset": int(choice_in_current_dataset),
                        "target_pool_size": len(target_cats),
                        "error": "",
                        "complex_raw_short": str(complex_raw)[:300].replace("\n", " "),
                        "agent_raw_short": str(agent_raw)[:300].replace("\n", " "),
                    })
                except Exception as e:
                    print(f"  failed GT category {gt_category_name}: {e}")
                    writer.writerow({
                        "image_id": img_id,
                        "file_name": file_name,
                        "status": "failed",
                        "gt_category_id": gt_category_id,
                        "gt_category_name": gt_category_name,
                        "complex_instruction": "",
                        "complex_status": "failed",
                        "direct_prompt": "",
                        "direct_raw_masks": 0,
                        "direct_top1_score": "",
                        "direct_top1_area": "",
                        "agent_status": "failed",
                        "agent_chosen_option_id": "",
                        "agent_chosen_category_id": "",
                        "agent_chosen_category_name": "",
                        "agent_prompt": "",
                        "agent_raw_masks": 0,
                        "agent_top1_score": "",
                        "agent_top1_area": "",
                        "agent_correct_category": "",
                        "agent_choice_in_current_dataset": "",
                        "target_pool_size": len(target_cats) if 'target_cats' in locals() else "",
                        "error": str(e),
                        "complex_raw_short": "",
                        "agent_raw_short": "",
                    })
                finally:
                    if idx % args.cache_save_every == 0:
                        save_cache(cache_file, api_cache)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if idx % args.flush_every == 0:
                with open(direct_pred_file, "w") as fp:
                    json.dump(direct_predictions, fp)
                with open(agent_pred_file, "w") as fp:
                    json.dump(agent_predictions, fp)
                save_cache(cache_file, api_cache)

    save_cache(cache_file, api_cache)
    with open(direct_pred_file, "w") as fp:
        json.dump(direct_predictions, fp)
    with open(agent_pred_file, "w") as fp:
        json.dump(agent_predictions, fp)

    direct_metrics = evaluate_prediction_file(eval_gt_file, direct_pred_file)
    agent_metrics = evaluate_prediction_file(eval_gt_file, agent_pred_file)
    delta = {k: agent_metrics.get(k, 0.0) - direct_metrics.get(k, 0.0) for k in METRIC_KEYS}

    print(f"Results for {dataset_key}:")
    print(f"  A. Complex instruction direct top1 : {direct_metrics}")
    print(f"  B. Agent parsed simple prompt top1 : {agent_metrics}")
    print(f"  Delta(B - A): {delta}")
    return {"direct_complex": direct_metrics, "agent_parsed": agent_metrics, "delta": delta}


def main():
    parser = argparse.ArgumentParser(
        "SAM3 complex-instruction front-agent eval: complex instruction direct vs API agent parsed prompt."
    )
    parser.add_argument("--config_path", default="/home/Data2/zhuquanhao/sam3/configs/internal_datasets/config.yaml")
    parser.add_argument("--checkpoint_path", default="/home/Data2/zhuquanhao/sam3/logs/internal_datasets_exp_v2/checkpoints/checkpoint_2.pt")
    parser.add_argument("--datasets_root", default="/home/Data2/zhuquanhao/datasets")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_images", type=int, default=0, help="Debug: evaluate first N images per dataset. 0 means all.")
    parser.add_argument("--subset_gt_for_max_images", action="store_true", default=True)
    parser.add_argument("--no_subset_gt_for_max_images", dest="subset_gt_for_max_images", action="store_false")
    parser.add_argument("--flush_every", type=int, default=20)
    parser.add_argument("--cache_save_every", type=int, default=10)

    parser.add_argument("--gt_category_mode", default="image_gt", choices=["image_gt", "first", "all_dataset"],
                        help="Which GT categories are used to synthesize complex instructions. image_gt is recommended.")
    parser.add_argument("--target_category_pool", default="organ_all", choices=["organ_all", "all_dataset", "current_dataset", "image_gt", "first"],
                        help="Which category list the agent can choose from. organ_all restricts choices to categories under the current organ/domain.")
    parser.add_argument("--fallback_simple_prompt_template", default="{category}",
                        help="Fallback prompt if agent parsing fails. Use {category} cleaned category or {raw_category} original name.")

    parser.add_argument("--api_key", default="sk-L250dA5cmSvKImoyZQv8LmjBcCXbhU2XGtCGlCD9MAeSTo9D")
    parser.add_argument("--api_url", default="https://api.bitidea.cn/v1/chat/completions")
    parser.add_argument("--api_model", default="gemini-3-pro-preview-11-2025")
    parser.add_argument("--api_timeout", type=int, default=180)
    parser.add_argument("--api_max_tokens", type=int, default=512)
    parser.add_argument("--api_image_max_side", type=int, default=768)
    parser.add_argument("--api_fallback_on_error", action="store_true", default=True)
    parser.add_argument("--no_api_fallback_on_error", dest="api_fallback_on_error", action="store_false")

    parser.add_argument("--complex_include_image", action="store_true",
                        help="Also send the image while generating synthetic complex instructions. Default is text-only to avoid leaking visual choice into instruction generation.")
    parser.add_argument("--complex_instruction_style", default="complex_but_unambiguous",
                        choices=["complex_but_unambiguous", "relational", "clinical", "layperson"],
                        help="Style hint for synthetic complex instruction generation.")
    parser.add_argument("--max_instruction_chars", type=int, default=300)

    parser.add_argument("--min_area", type=int, default=1)
    parser.add_argument("--max_area_ratio", type=float, default=1.0)
    parser.add_argument("--remove_overlap", action="store_true", help="Apply SAM3 overlap removal before top1 selection.")

    args = parser.parse_args()
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"/home/Data2/zhuquanhao/sam3/logs/eval_sam3_complex_agent_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config not found: {args.config_path}")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    model, device = setup_model(args.config_path, args.checkpoint_path)
    test_datasets = get_all_datasets(args.datasets_root)
    print(f"Found {len(test_datasets)} datasets to test.")
    args.organ_category_pools = build_organ_category_pools(test_datasets)
    if args.target_category_pool == "organ_all":
        print("Using organ-level category pools. The agent will choose only from categories under the current organ/domain.")
        for organ, pool in sorted(args.organ_category_pools.items()):
            print(f"  {organ}: {len(pool)} unique category meanings")
    print(f"Only metrics: {', '.join(METRIC_KEYS)}")
    print("A: complex user instruction direct -> SAM3 top1")
    print("B: API agent sees image + instruction + category list -> simple SAM3 prompt -> SAM3 top1")

    results = {}
    for dataset_key, dataset_info in test_datasets.items():
        results[dataset_key] = run_dataset(dataset_key, dataset_info, model, device, args)

    summary_file = os.path.join(args.output_dir, "summary.csv")
    print("\n" + "=" * 160)
    print("Summary columns: A direct complex, B agent parsed, delta B-A for each metric")
    print("=" * 160)
    with open(summary_file, "w") as f:
        header = [
            "Organ", "Dataset",
            "A_DirectComplex_mAP(0.50:0.95)", "B_AgentParsed_mAP(0.50:0.95)", "Delta_mAP",
            "A_DirectComplex_AP(0.50)", "B_AgentParsed_AP(0.50)", "Delta_AP50",
            "A_DirectComplex_Mean_IoU", "B_AgentParsed_Mean_IoU", "Delta_IoU",
            "A_DirectComplex_Mean_Dice", "B_AgentParsed_Mean_Dice", "Delta_Dice",
        ]
        f.write(",".join(header) + "\n")
        current_organ = ""
        for key in sorted(test_datasets.keys()):
            organ, dataset = key.split("/")
            if organ != current_organ:
                print(f"-- {organ} --")
                current_organ = organ
            res = results.get(key)
            if res is None:
                continue
            def fmt(v):
                return f"{v:.4f}" if v is not None else "-"
            direct = res.get("direct_complex", {})
            agent = res.get("agent_parsed", {})
            delta = res.get("delta", {})
            vals = [
                direct.get("mAP (0.50:0.95)", 0.0), agent.get("mAP (0.50:0.95)", 0.0), delta.get("mAP (0.50:0.95)", 0.0),
                direct.get("AP (0.50)", 0.0), agent.get("AP (0.50)", 0.0), delta.get("AP (0.50)", 0.0),
                direct.get("Mean IoU", 0.0), agent.get("Mean IoU", 0.0), delta.get("Mean IoU", 0.0),
                direct.get("Mean Dice", 0.0), agent.get("Mean Dice", 0.0), delta.get("Mean Dice", 0.0),
            ]
            print(
                f"{dataset:<40} | "
                f"mAP {fmt(vals[0])}->{fmt(vals[1])} ({fmt(vals[2])}) | "
                f"AP50 {fmt(vals[3])}->{fmt(vals[4])} ({fmt(vals[5])}) | "
                f"IoU {fmt(vals[6])}->{fmt(vals[7])} ({fmt(vals[8])}) | "
                f"Dice {fmt(vals[9])}->{fmt(vals[10])} ({fmt(vals[11])})"
            )
            f.write(f"{organ},{dataset}," + ",".join(fmt(v) for v in vals) + "\n")
    print(f"\nEvaluation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
