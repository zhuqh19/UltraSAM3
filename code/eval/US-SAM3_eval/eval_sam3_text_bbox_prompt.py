#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate US-SAM3 / SAM3 with BOTH text prompt and dataset GT bbox prompt.

Prompt protocol:
    state = processor.set_image(image)
    processor.set_text_prompt(state=state, prompt=category_name)
    output = processor.add_geometric_prompt(state=state, box=normalized_cxcywh, label=True)

Notes:
- This is an oracle / privileged prompt setting because it uses GT bbox from the dataset.
- Each GT annotation bbox is used as a positive box prompt together with its category text.
- COCO AP is computed on segmentation masks.
- Mean IoU / Dice are computed from dumped COCO predictions by image-level foreground union.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from PIL import Image
import cv2
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils


DEFAULT_SAM3_CODE_DIR = "/home/Data2/zhuquanhao/sam3/code"
DEFAULT_CHECKPOINT = "/home/Data2/zhuquanhao/sam3/weight/sam3.pt"
DEFAULT_DATASETS_ROOT = "/home/Data2/zhuquanhao/datasets"
DEFAULT_OUTPUT_ROOT = "/home/Data2/zhuquanhao/sam3/logs"
DEFAULT_BPE_PATH = "/home/Data2/zhuquanhao/sam3/code/assets/bpe_simple_vocab_16e6.txt.gz"


def get_all_datasets(datasets_root):
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
            organ_path_name = organ
            if organ == "Fetal":
                organ_path_name = "fetal"
            if organ == "Liver":
                organ_path_name = "liver"
            if organ == "Muscle":
                organ_path_name = "muscle"
            test_datasets[key] = {
                "name": dataset_name,
                "img_folder": f"{datasets_root}/{organ_path_name}/Datasets/{dataset_name}/test/",
                "ann_file": f"{datasets_root}/{organ_path_name}/Datasets/{dataset_name}/test/_annotations.coco.json",
            }
    return test_datasets


def load_sam3_processor(args):
    sys.path.insert(0, args.sam3_code_dir)
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print(f"[INFO] Loading SAM3/US-SAM3 on {args.device}")
    print(f"[INFO] Checkpoint: {args.checkpoint_path}")

    build_kwargs = {
        "bpe_path": args.bpe_path,
        "eval_mode": True,
        "device": args.device,
    }
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        build_kwargs.update({
            "checkpoint_path": args.checkpoint_path,
            "load_from_HF": False,
        })

    try:
        model = build_sam3_image_model(**build_kwargs)
    except TypeError:
        print("[WARN] build_sam3_image_model did not accept all kwargs; retrying with minimal args.")
        model = build_sam3_image_model(
            checkpoint_path=args.checkpoint_path,
            bpe_path=args.bpe_path,
            load_from_HF=False,
        )
        model = model.to(args.device).eval()
    except Exception as e:
        print(f"[WARN] Direct checkpoint loading failed: {e}")
        print("[WARN] Retrying by building model and loading state_dict manually.")
        model = build_sam3_image_model(
            bpe_path=args.bpe_path,
            load_from_HF=False,
            device=args.device,
            eval_mode=True,
        )
        ckpt = torch.load(args.checkpoint_path, map_location="cpu")
        state = ckpt
        for key in ["model", "state_dict", "model_state_dict"]:
            if isinstance(ckpt, dict) and key in ckpt:
                state = ckpt[key]
                break
        new_state = {}
        for k, v in state.items():
            nk = k
            for prefix in ["module.", "model."]:
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            new_state[nk] = v
        msg = model.load_state_dict(new_state, strict=False)
        print(f"[INFO] Manual load_state_dict result: {msg}")
        model = model.to(args.device).eval()

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold)
    return processor


def coco_xywh_to_norm_cxcywh(bbox_xywh, img_w, img_h):
    x, y, w, h = [float(v) for v in bbox_xywh]
    return [(x + w / 2.0) / img_w, (y + h / 2.0) / img_h, w / img_w, h / img_h]


def xywh_to_xyxy(bbox_xywh):
    x, y, w, h = [float(v) for v in bbox_xywh]
    return [x, y, x + w, y + h]


def norm_cxcywh_to_xyxy(box, img_w, img_h):
    cx, cy, w, h = [float(v) for v in box]
    cx *= img_w
    cy *= img_h
    w *= img_w
    h *= img_h
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def box_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def extract_output_dict(output, state):
    if isinstance(output, dict) and "masks" in output:
        return output
    if isinstance(state, dict) and "masks" in state:
        return state
    if isinstance(output, dict):
        for k in ["outputs", "predictions"]:
            if k in output and isinstance(output[k], dict) and "masks" in output[k]:
                return output[k]
    return output if isinstance(output, dict) else {}


def run_text_bbox_prompt(processor, image_pil, text_prompt, norm_box_cxcywh):
    state = processor.set_image(image_pil)

    # Set text first. In official SAM3, add_geometric_prompt then uses the existing language features.
    text_output = processor.set_text_prompt(state=state, prompt=text_prompt)
    if isinstance(text_output, dict) and "backbone_out" in text_output:
        state = text_output

    try:
        output = processor.add_geometric_prompt(state=state, box=norm_box_cxcywh, label=True)
    except TypeError:
        output = processor.add_geometric_prompt(norm_box_cxcywh, True, state)

    return extract_output_dict(output, state)


def select_one_output(output, prompt_bbox_xywh, img_w, img_h, selection_mode, mask_threshold):
    masks = to_numpy(output.get("masks"))
    scores = to_numpy(output.get("scores"))
    boxes = to_numpy(output.get("boxes"))

    if masks is None or len(masks) == 0:
        return None

    if masks.ndim == 2:
        masks = masks[None, ...]
    elif masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    n = masks.shape[0]
    if scores is None or len(scores) == 0:
        scores = np.ones((n,), dtype=np.float32)
    scores = np.asarray(scores).reshape(-1)[:n]

    prompt_xyxy = xywh_to_xyxy(prompt_bbox_xywh)
    pred_box_ious = np.zeros((n,), dtype=np.float32)

    if boxes is not None and len(boxes) > 0:
        boxes = np.asarray(boxes)
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        boxes = boxes[:n]
        for i in range(len(boxes)):
            b = boxes[i].tolist()
            if max(b) <= 1.5:
                pred_xyxy = norm_cxcywh_to_xyxy(b, img_w, img_h)
            else:
                pred_xyxy = b[:4]
            pred_box_ious[i] = box_iou_xyxy(pred_xyxy, prompt_xyxy)

    if selection_mode == "top_score":
        idx = int(np.argmax(scores))
    elif selection_mode == "bbox_iou":
        idx = int(np.argmax(pred_box_ious))
    elif selection_mode == "bbox_score":
        idx = int(np.argmax(scores * (1.0 + pred_box_ious)))
    else:
        raise ValueError(f"Unknown selection mode: {selection_mode}")

    score = float(scores[idx])
    mask = masks[idx]
    if mask.shape != (img_h, img_w):
        mask = cv2.resize(mask.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_LINEAR)

    binary = (mask > mask_threshold).astype(np.uint8)
    if binary.sum() == 0:
        return None

    return {
        "mask": binary,
        "score": score,
        "bbox_iou": float(pred_box_ious[idx]) if len(pred_box_ious) > idx else 0.0,
    }


def mask_to_coco_pred(mask, image_id, category_id, score, pred_id):
    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        return None

    rle = maskUtils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("utf-8")

    ys, xs = np.where(mask > 0)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    bbox = [float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1)]

    return {
        "id": int(pred_id),
        "image_id": int(image_id),
        "category_id": int(category_id),
        "segmentation": rle,
        "bbox": bbox,
        "area": float(mask.sum()),
        "score": float(score),
    }


def compute_iou_dice_from_coco(gt_file, pred_file):
    if not os.path.exists(pred_file):
        return 0.0, 0.0

    coco_gt = COCO(gt_file)
    with open(pred_file, "r") as f:
        preds = json.load(f)
    if not preds:
        return 0.0, 0.0

    coco_dt = coco_gt.loadRes(preds)
    ious, dices = [], []

    for img_id in coco_gt.getImgIds():
        img_info = coco_gt.loadImgs(img_id)[0]
        h, w = img_info["height"], img_info["width"]

        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        if len(ann_ids) == 0:
            continue

        gt_mask = np.zeros((h, w), dtype=np.uint8)
        for ann in coco_gt.loadAnns(ann_ids):
            gt_mask = np.maximum(gt_mask, coco_gt.annToMask(ann).astype(np.uint8))

        pred_mask = np.zeros((h, w), dtype=np.uint8)
        dt_ann_ids = coco_dt.getAnnIds(imgIds=img_id)
        for ann in coco_dt.loadAnns(dt_ann_ids):
            pred_mask = np.maximum(pred_mask, coco_dt.annToMask(ann).astype(np.uint8))

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        denom = pred_mask.sum() + gt_mask.sum()

        if union == 0:
            iou, dice = 0.0, 0.0
        else:
            iou = intersection / union
            dice = 0.0 if denom == 0 else 2.0 * intersection / denom

        ious.append(iou)
        dices.append(dice)

    return float(np.mean(ious)) if ious else 0.0, float(np.mean(dices)) if dices else 0.0


def evaluate_coco(gt_file, pred_file):
    coco_gt = COCO(gt_file)
    with open(pred_file, "r") as f:
        preds = json.load(f)

    if not preds:
        return {
            "mAP (0.50:0.95)": 0.0,
            "AP (0.50)": 0.0,
            "Mean IoU": 0.0,
            "Mean Dice": 0.0,
        }

    coco_dt = coco_gt.loadRes(preds)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mean_iou, mean_dice = compute_iou_dice_from_coco(gt_file, pred_file)

    return {
        "mAP (0.50:0.95)": float(coco_eval.stats[0]),
        "AP (0.50)": float(coco_eval.stats[1]),
        "Mean IoU": mean_iou,
        "Mean Dice": mean_dice,
    }


def evaluate_dataset(processor, dataset_key, dataset_info, output_root, args):
    print(f"\n{'='*80}")
    print(f"Evaluating {dataset_key} with text + GT bbox prompts")
    print(f"{'='*80}")

    img_folder = dataset_info["img_folder"]
    ann_file = dataset_info["ann_file"]
    if not os.path.exists(ann_file):
        print(f"[WARN] Annotation file not found: {ann_file}")
        return None

    coco = COCO(ann_file)
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    out_dir = Path(output_root) / dataset_key.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / "coco_predictions_text_bbox_segm.json"

    predictions = []
    pred_id = 1
    num_prompted = 0
    num_kept = 0
    bbox_iou_scores = []

    for img_id in tqdm(coco.getImgIds(), desc=dataset_key):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_folder, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        coco_w, coco_h = int(img_info["width"]), int(img_info["height"])
        if image.size != (coco_w, coco_h):
            image = image.resize((coco_w, coco_h), Image.Resampling.BILINEAR)

        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            cat_id = int(ann["category_id"])
            cat_name = cats.get(cat_id, "object")
            bbox_xywh = ann.get("bbox")
            if bbox_xywh is None:
                continue

            norm_box = coco_xywh_to_norm_cxcywh(bbox_xywh, coco_w, coco_h)

            try:
                output = run_text_bbox_prompt(processor, image, cat_name, norm_box)
                selected = select_one_output(
                    output=output,
                    prompt_bbox_xywh=bbox_xywh,
                    img_w=coco_w,
                    img_h=coco_h,
                    selection_mode=args.selection_mode,
                    mask_threshold=args.mask_threshold,
                )
            except Exception as e:
                if args.debug:
                    print(f"[WARN] Failed image={img_id}, cat={cat_name}, bbox={bbox_xywh}: {repr(e)}")
                selected = None

            num_prompted += 1
            if selected is None or selected["score"] < args.min_score_to_keep:
                continue

            pred = mask_to_coco_pred(
                mask=selected["mask"],
                image_id=img_id,
                category_id=cat_id,
                score=selected["score"],
                pred_id=pred_id,
            )
            if pred is None:
                continue

            predictions.append(pred)
            pred_id += 1
            num_kept += 1
            bbox_iou_scores.append(selected.get("bbox_iou", 0.0))

            if args.empty_cache_every > 0 and num_prompted % args.empty_cache_every == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    with open(pred_file, "w") as f:
        json.dump(predictions, f)

    print(f"[INFO] Prompted GT boxes: {num_prompted}")
    print(f"[INFO] Kept predictions: {num_kept}")
    if bbox_iou_scores:
        print(f"[INFO] Mean selected bbox IoU with prompt bbox: {np.mean(bbox_iou_scores):.4f}")

    metrics = evaluate_coco(ann_file, str(pred_file))
    metrics["Prompted Boxes"] = num_prompted
    metrics["Kept Predictions"] = num_kept
    metrics["Mean Prompt-BBox IoU"] = float(np.mean(bbox_iou_scores)) if bbox_iou_scores else 0.0

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[RESULT] {dataset_key}: {metrics}")
    return metrics


def write_summary(output_root, results):
    summary_file = Path(output_root) / "summary.csv"
    with open(summary_file, "w") as f:
        f.write("Organ,Dataset,mAP(0.50:0.95),AP(0.50),Mean IoU,Mean Dice,Prompted Boxes,Kept Predictions,Mean Prompt-BBox IoU\n")
        for key in sorted(results.keys()):
            metrics = results[key]
            if not metrics:
                continue
            organ, dataset = key.split("/")
            f.write(
                f"{organ},{dataset},"
                f"{metrics.get('mAP (0.50:0.95)', 0.0):.4f},"
                f"{metrics.get('AP (0.50)', 0.0):.4f},"
                f"{metrics.get('Mean IoU', 0.0):.4f},"
                f"{metrics.get('Mean Dice', 0.0):.4f},"
                f"{metrics.get('Prompted Boxes', 0)},"
                f"{metrics.get('Kept Predictions', 0)},"
                f"{metrics.get('Mean Prompt-BBox IoU', 0.0):.4f}\n"
            )
    print(f"[INFO] Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser("Evaluate SAM3 / US-SAM3 with text + GT bbox prompts")

    parser.add_argument("--sam3-code-dir", type=str, default=DEFAULT_SAM3_CODE_DIR)
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--bpe-path", type=str, default=DEFAULT_BPE_PATH)
    parser.add_argument("--datasets-root", type=str, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)

    parser.add_argument(
        "--selection-mode",
        type=str,
        default="bbox_score",
        choices=["top_score", "bbox_iou", "bbox_score"],
        help="How to select one output mask from SAM3 for each text+bbox prompt.",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-score-to-keep", type=float, default=0.0)
    parser.add_argument("--empty-cache-every", type=int, default=200)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--only-dataset",
        type=str,
        default=None,
        help="Optional dataset key, e.g., Cardiac/CAMUS_coco, for debugging.",
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = os.path.join(args.output_root, f"eval_text_bbox_prompt_{args.selection_mode}_{timestamp}")
    os.makedirs(output_root, exist_ok=True)

    processor = load_sam3_processor(args)

    datasets = get_all_datasets(args.datasets_root)
    if args.only_dataset:
        datasets = {k: v for k, v in datasets.items() if k == args.only_dataset}
        if not datasets:
            raise ValueError(f"Dataset key not found: {args.only_dataset}")

    print(f"[INFO] Found {len(datasets)} datasets to evaluate.")
    print(f"[INFO] Output root: {output_root}")
    print("[INFO] Prompt mode: text + GT bbox")
    print(f"[INFO] Selection mode: {args.selection_mode}")

    results = {}
    for key, info in datasets.items():
        metrics = evaluate_dataset(processor, key, info, output_root, args)
        results[key] = metrics
        write_summary(output_root, results)

    print("\n" + "=" * 120)
    print(f"{'Dataset':<40} | {'mAP':<10} | {'AP50':<10} | {'mIoU':<10} | {'mDice':<10} | {'Kept/Prompted':<15}")
    print("-" * 120)
    for key in sorted(results.keys()):
        m = results[key]
        if not m:
            continue
        kept = m.get("Kept Predictions", 0)
        prompted = m.get("Prompted Boxes", 0)
        print(
            f"{key:<40} | "
            f"{m.get('mAP (0.50:0.95)', 0.0):<10.4f} | "
            f"{m.get('AP (0.50)', 0.0):<10.4f} | "
            f"{m.get('Mean IoU', 0.0):<10.4f} | "
            f"{m.get('Mean Dice', 0.0):<10.4f} | "
            f"{kept}/{prompted:<15}"
        )

    print(f"\nEvaluation complete. Results saved to {output_root}")


if __name__ == "__main__":
    main()
