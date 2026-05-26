#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-image question -> agent-parsed SAM3 prompt -> top1 segmentation mask.

  1) load SAM3-family checkpoint with Hydra config;
  2) send image + natural-language question to a front-end API agent;
  3) parse the agent JSON response and obtain one short `sam3_prompt`;
  4) run direct SAM3 inference with collate_fn_api + model(batch_input);
  5) select the top-1 candidate by score;
  6) save binary mask, overlay, candidates JSON, and meta JSON.

Typical usage:
  python inference_agent.py \
    --image /path/to/case002.png \
    --question "Please segment the thyroid nodule in this ultrasound image." \
    --api-key $OPENAI_API_KEY
    --api-url your_api_url

If no API key is available, use --no-api to fall back to a simple local prompt
rewrite based on the question text.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import base64
import gc
import json
import re
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import pycocotools.mask as maskUtils


# ========================= Default local paths =========================
# Modify these three defaults if your code/config/checkpoint locations differ.
DEFAULT_SAM3_CODE_DIR = "../code"
DEFAULT_CONFIG_PATH = "../config/config.yaml"
DEFAULT_CHECKPOINT_PATH = "/home/Data2/zhuquanhao/US-SAM3/US-SAM3_weight/US-SAM3.pt"
DEFAULT_OUTPUT_ROOT = "./sam3_agent_single_image_outputs"

# A small built-in ultrasound category pool. You can override with --categories.
DEFAULT_CATEGORY_OPTIONS = [
    "thyroid nodule",
    "thyroid gland",
    "breast lesion",
    "tumor",
    "lesion",
    "liver tumor",
    "kidney",
    "common carotid artery",
    "carotid intima-media region",
    "left ventricular cavity",
    "left atrium",
    "right ventricle",
    "right atrium",
    "myocardium",
    "fetal head",
    "fetal abdomen",
    "prostate",
    "nerve",
    "biceps brachii muscle",
    "gastrocnemius muscle",
    "tibialis anterior muscle",
]

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


# ========================= Utility functions =========================
def clean_category_prompt(category_name: str) -> str:
    if category_name in CATEGORY_PROMPT_ALIASES:
        return CATEGORY_PROMPT_ALIASES[category_name]
    s = str(category_name).strip()
    s = CATEGORY_PROMPT_ALIASES.get(s, s)
    s = s.replace("_", " ").replace("-", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s or "object"


def parse_categories(categories_arg: Optional[str], categories_file: Optional[str]) -> List[Dict[str, Any]]:
    """Return a fixed list of valid target categories for the front agent."""
    raw: List[str] = []
    if categories_file:
        p = Path(categories_file).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"categories file not found: {p}")
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".json":
            obj = json.loads(text)
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, str):
                        raw.append(item)
                    elif isinstance(item, dict):
                        raw.append(str(item.get("meaning") or item.get("name") or item.get("category") or ""))
        else:
            raw.extend([x.strip() for x in re.split(r"[,\n]", text) if x.strip()])
    if categories_arg:
        raw.extend([x.strip() for x in categories_arg.split(",") if x.strip()])
    if not raw:
        raw = list(DEFAULT_CATEGORY_OPTIONS)

    # De-duplicate by cleaned meaning, keep a stable option_id.
    seen = set()
    out = []
    for name in raw:
        meaning = clean_category_prompt(name)
        key = meaning.lower()
        if not meaning or key in seen:
            continue
        seen.add(key)
        out.append({
            "option_id": str(len(out) + 1),
            "name": name,
            "meaning": meaning,
        })
    return out


def load_image_as_rgb(image_path: str) -> Image.Image:
    with Image.open(image_path) as im:
        im.load()
        return im.convert("RGB")


def pil_to_data_url(img: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    buf = BytesIO()
    img = img.convert("RGB")
    img.save(buf, format=fmt, quality=quality, optimize=True)
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def image_path_to_data_url(image_path: str, max_side: int = 768) -> str:
    with Image.open(image_path) as im:
        im.load()
        im = im.convert("RGB")
        if max_side and max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.BILINEAR)
        return pil_to_data_url(im, fmt="JPEG", quality=90)


def extract_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def colorize_overlay(image_rgb: Image.Image, mask: np.ndarray, alpha: float = 0.45, color=(255, 0, 0)) -> Image.Image:
    arr = np.asarray(image_rgb.convert("RGB")).copy()
    mask_bool = (mask > 0)
    overlay = arr.copy()
    overlay[mask_bool] = (np.array(color) * alpha + overlay[mask_bool] * (1 - alpha)).astype(np.uint8)
    return Image.fromarray(overlay)


def decode_rle_mask(rle: dict, h: int, w: int) -> np.ndarray:
    if isinstance(rle, dict) and isinstance(rle.get("counts"), list):
        rle = maskUtils.frPyObjects(rle, h, w)
    m = maskUtils.decode(rle)
    if m.ndim == 3:
        m = np.max(m, axis=2)
    if m.shape[:2] != (h, w):
        import cv2
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


# ========================= SAM3 model + direct inference path =========================
def add_sam3_to_path(sam3_code_dir: str):
    sam3_code_dir = os.path.abspath(os.path.expanduser(sam3_code_dir))
    if sam3_code_dir not in sys.path:
        sys.path.insert(0, sam3_code_dir)


def setup_model(config_path: str, checkpoint_path: str, sam3_code_dir: str):
    """Adapted from eval_agent_all_datasets_v2.py::setup_model."""
    add_sam3_to_path(sam3_code_dir)
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from sam3.train.utils.train_utils import register_omegaconf_resolvers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        register_omegaconf_resolvers()
    except Exception:
        pass

    config_dir = os.path.dirname(os.path.abspath(config_path))
    config_name = os.path.basename(config_path)
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=config_dir, version_base="1.2")
    cfg = compose(config_name=config_name)
    cfg.trainer.model.checkpoint_path = None
    cfg.trainer.model.load_from_HF = False

    print("[INFO] Loading SAM3 model...")
    model = instantiate(cfg.trainer.model)
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "model_state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    # Strip common distributed/trainer prefixes if present.
    if isinstance(ckpt, dict):
        stripped = {}
        for k, v in ckpt.items():
            nk = k
            for prefix in ["module.", "model.", "_orig_mod."]:
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            stripped[nk] = v
        ckpt = stripped
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"[INFO] Loaded checkpoint with missing={len(missing)}, unexpected={len(unexpected)}")
    model.to(device)
    model.eval()
    return model, device, {"missing_keys": len(missing), "unexpected_keys": len(unexpected)}


def build_batch_for_sam3(image_pil: Image.Image, text_prompt: str, device: str):
    """Adapted from eval_agent_all_datasets_v2.py::build_batch_for_sam3."""
    from sam3.train.data.collator import collate_fn_api
    from sam3.train.data.sam3_image_dataset import (
        Datapoint,
        Image as SAMImage,
        FindQueryLoaded,
        InferenceMetadata,
    )

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


def extract_logits_masks(outputs, device: str):
    """Adapted from eval_agent_all_datasets_v2.py::extract_logits_masks."""
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


def masks_to_serialized(pred_logits, pred_masks, orig_h: int, orig_w: int) -> Dict[str, Any]:
    """Adapted from eval_agent_all_datasets_v2.py::masks_to_serialized."""
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


def maybe_remove_overlaps(serialized: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from sam3.agent.helpers.mask_overlap_removal import remove_overlapping_masks
    except Exception:
        return serialized
    h, w = serialized["orig_img_h"], serialized["orig_img_w"]
    tmp = {
        "orig_img_h": h,
        "orig_img_w": w,
        "pred_boxes": serialized.get("pred_boxes", []),
        "pred_masks": [r["counts"] if isinstance(r, dict) else r for r in serialized.get("pred_masks", [])],
        "pred_scores": serialized.get("pred_scores", []),
    }
    tmp = remove_overlapping_masks(tmp)
    masks, areas = [], []
    for counts in tmp.get("pred_masks", []):
        rle = {"size": [h, w], "counts": counts}
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


def sam3_predict_image(model, device: str, image_path: str, text_prompt: str, remove_overlap: bool = False) -> Dict[str, Any]:
    image_pil = load_image_as_rgb(image_path)
    batch_input, orig_h, orig_w = build_batch_for_sam3(image_pil, text_prompt, device)
    try:
        with torch.inference_mode():
            outputs = model(batch_input)
        pred_logits, pred_masks = extract_logits_masks(outputs, device)
        serialized = masks_to_serialized(pred_logits, pred_masks, orig_h, orig_w)
        if remove_overlap:
            serialized = maybe_remove_overlaps(serialized)
        return serialized
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def collect_candidates(serialized: Dict[str, Any], prompt: str, min_area: int, max_area_ratio: float) -> List[Dict[str, Any]]:
    candidates = []
    h, w = int(serialized["orig_img_h"]), int(serialized["orig_img_w"])
    max_area = int(h * w * max_area_ratio) if max_area_ratio and max_area_ratio > 0 else None
    masks = serialized.get("pred_masks", [])
    scores = serialized.get("pred_scores", [0.0] * len(masks))
    areas = serialized.get("pred_areas", [0] * len(masks))
    boxes = serialized.get("pred_boxes", [[0, 0, 0, 0]] * len(masks))
    for mask_idx, rle in enumerate(masks):
        area = int(areas[mask_idx])
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        candidates.append({
            "prompt": prompt,
            "mask_idx": int(mask_idx),
            "score": float(scores[mask_idx]),
            "area": area,
            "box": boxes[mask_idx],
            "rle": rle,
            "h": h,
            "w": w,
        })
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


# ========================= Front-end agent =========================
def call_chat_api(messages: List[Dict[str, Any]], args, max_tokens: Optional[int] = None) -> str:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"}
    payload = {
        "model": args.api_model,
        "messages": messages,
        "temperature": args.api_temperature,
        "max_tokens": max_tokens or args.api_max_tokens,
    }
    resp = requests.post(args.api_url, headers=headers, json=payload, timeout=args.api_timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def local_fallback_prompt(question: str, categories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple no-API fallback: choose category by substring, otherwise clean question."""
    q = question.lower()
    best = None
    for c in categories:
        meaning = c["meaning"].lower()
        name = str(c["name"]).lower()
        if meaning in q or name in q:
            best = c
            break
    if best is not None:
        prompt = best["meaning"]
        return {
            "chosen_option_id": best["option_id"],
            "chosen_category_name": best["name"],
            "sam3_prompt": prompt,
            "reason": "local fallback matched category text in question",
            "status": "local_fallback_category_match",
        }
    # Remove common instruction words; keep a short noun-ish phrase.
    prompt = re.sub(r"(?i)\b(please|segment|identify|find|show|mask|outline|this|the|in|image|ultrasound|us|for|me)\b", " ", question)
    prompt = re.sub(r"[^A-Za-z0-9\-_/ ]+", " ", prompt)
    prompt = re.sub(r"\s+", " ", prompt).strip()
    if not prompt:
        prompt = question.strip()
    prompt = " ".join(prompt.split()[:10])
    return {
        "chosen_option_id": "free_fallback",
        "chosen_category_name": prompt,
        "sam3_prompt": prompt,
        "reason": "local fallback cleaned the question because no category matched",
        "status": "local_fallback_clean_question",
    }


def agent_parse_question(image_path: str, question: str, categories: List[Dict[str, Any]], args) -> Tuple[Dict[str, Any], str]:
    if args.no_api:
        parsed = local_fallback_prompt(question, categories)
        return parsed, ""
    if not args.api_key:
        parsed = local_fallback_prompt(question, categories)
        parsed["status"] = "local_fallback_missing_api_key"
        return parsed, ""

    target_payload = [
        {
            "option_id": c["option_id"],
            "name": c["name"],
            "meaning": c["meaning"],
        }
        for c in categories
    ]
    system_text = (
        "You are a lightweight front-end agent for SAM3 ultrasound segmentation. "
        "You receive an image, a user's natural-language segmentation question, "
        "and an optional fixed list of valid ultrasound target categories. "
        "Your job is not to segment masks. Your job is to rewrite the user question "
        "into one concise SAM3 text prompt that directly names the target anatomy/pathology. "
        "If the category list contains the target, choose exactly one option from it. "
        "If none clearly match, still produce the best short SAM3 prompt. "
        "Return ONLY valid JSON with keys: chosen_option_id, chosen_category_name, sam3_prompt, reason."
    )
    user_text = (
        f"User segmentation question: {question!r}\n\n"
        "Valid target categories as JSON array:\n"
        f"{json.dumps(target_payload, ensure_ascii=False)}\n\n"
        "Write sam3_prompt as a concise noun phrase or very short instruction under 14 words. "
        "Prefer concrete medical/ultrasound terms like 'thyroid nodule' instead of vague words like 'it'."
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
        obj = extract_json_object(raw)
        if not isinstance(obj, dict):
            parsed = local_fallback_prompt(question, categories)
            parsed["status"] = "local_fallback_invalid_agent_json"
            parsed["raw_agent_response"] = raw[:1000]
            return parsed, raw

        prompt = str(obj.get("sam3_prompt") or obj.get("prompt") or obj.get("simple_prompt") or "").strip()
        chosen_option = str(obj.get("chosen_option_id") or obj.get("option_id") or "").strip()
        chosen_name = str(obj.get("chosen_category_name") or obj.get("category_name") or "").strip()
        reason = str(obj.get("reason", ""))

        by_option = {c["option_id"]: c for c in categories}
        by_name = {str(c["name"]).strip().lower(): c for c in categories}
        by_meaning = {str(c["meaning"]).strip().lower(): c for c in categories}
        chosen_cat = by_option.get(chosen_option)
        if chosen_cat is None and chosen_name:
            chosen_cat = by_name.get(chosen_name.lower()) or by_meaning.get(clean_category_prompt(chosen_name).lower())
        if not prompt and chosen_cat is not None:
            prompt = chosen_cat["meaning"]
        if not prompt:
            fallback = local_fallback_prompt(question, categories)
            prompt = fallback["sam3_prompt"]

        parsed = {
            "chosen_option_id": chosen_cat["option_id"] if chosen_cat is not None else (chosen_option or "free_text"),
            "chosen_category_name": chosen_cat["name"] if chosen_cat is not None else (chosen_name or prompt),
            "sam3_prompt": re.sub(r"\s+", " ", prompt).strip(),
            "reason": reason,
            "status": "api",
        }
        return parsed, raw
    except Exception as e:
        if not args.api_fallback_on_error:
            raise
        parsed = local_fallback_prompt(question, categories)
        parsed["status"] = "local_fallback_api_error"
        parsed["error"] = str(e)
        return parsed, f"ERROR: {e}"


# ========================= Save outputs =========================
def save_outputs(
    image_path: str,
    question: str,
    parsed: Dict[str, Any],
    raw_agent_response: str,
    candidates: List[Dict[str, Any]],
    serialized: Dict[str, Any],
    args,
    load_info: Dict[str, Any],
) -> Dict[str, str]:
    stem = Path(image_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root).expanduser().resolve() / f"{stem}_agent_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    image = load_image_as_rgb(image_path)
    h, w = int(serialized["orig_img_h"]), int(serialized["orig_img_w"])
    if image.size != (w, h):
        image = image.resize((w, h), Image.BILINEAR)

    top1 = candidates[0] if candidates else None
    if top1 is None:
        mask = np.zeros((h, w), dtype=np.uint8)
    else:
        mask = decode_rle_mask(top1["rle"], h, w)

    mask_path = run_dir / f"{stem}_agent_top1_mask.png"
    overlay_path = run_dir / f"{stem}_agent_top1_overlay.png"
    candidates_path = run_dir / f"{stem}_agent_candidates.json"
    meta_path = run_dir / f"{stem}_agent_meta.json"
    raw_agent_path = run_dir / f"{stem}_agent_raw_response.txt"

    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(mask_path)
    colorize_overlay(image, mask).save(overlay_path)

    json_safe_candidates = []
    for c in candidates:
        item = dict(c)
        item["rle"] = dict(item["rle"])
        if isinstance(item["rle"].get("counts"), bytes):
            item["rle"]["counts"] = item["rle"]["counts"].decode("utf-8")
        json_safe_candidates.append(item)
    candidates_path.write_text(json.dumps(json_safe_candidates, indent=2, ensure_ascii=False), encoding="utf-8")
    raw_agent_path.write_text(raw_agent_response or "", encoding="utf-8")

    meta = {
        "image": str(Path(image_path).expanduser().resolve()),
        "question": question,
        "agent_status": parsed.get("status", ""),
        "agent_chosen_option_id": parsed.get("chosen_option_id", ""),
        "agent_chosen_category_name": parsed.get("chosen_category_name", ""),
        "sam3_prompt": parsed.get("sam3_prompt", ""),
        "agent_reason": parsed.get("reason", ""),
        "checkpoint": str(Path(args.checkpoint_path).expanduser().resolve()),
        "config_path": str(Path(args.config_path).expanduser().resolve()),
        "sam3_code_dir": str(Path(args.sam3_code_dir).expanduser().resolve()),
        "num_raw_masks": len(serialized.get("pred_masks", [])),
        "num_candidates_after_filter": len(candidates),
        "top1_score": None if top1 is None else float(top1["score"]),
        "top1_area_pixels": int(mask.sum()),
        "top1_box": None if top1 is None else top1.get("box"),
        "min_area": args.min_area,
        "max_area_ratio": args.max_area_ratio,
        "remove_overlap": bool(args.remove_overlap),
        "model_load_info": load_info,
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "candidates_path": str(candidates_path),
        "raw_agent_response_path": str(raw_agent_path),
        "run_dir": str(run_dir),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if top1 is None:
        print("[WARN] No valid candidate after filtering. Saved an empty mask.")
    else:
        print(f"[INFO] Agent SAM3 prompt: {meta['sam3_prompt']!r}")
        print(f"[INFO] Raw masks={meta['num_raw_masks']}, candidates={len(candidates)}, top1 score={meta['top1_score']:.6f}, area={meta['top1_area_pixels']} pixels")

    return {
        "run_dir": str(run_dir),
        "mask": str(mask_path),
        "overlay": str(overlay_path),
        "meta": str(meta_path),
        "candidates": str(candidates_path),
        "raw_agent_response": str(raw_agent_path),
    }


# ========================= CLI =========================
def parse_args():
    parser = argparse.ArgumentParser("Single-image SAM3 agent inference: question -> prompt -> top1 mask")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--question", required=True, help="Natural-language segmentation question/instruction")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument("--sam3-code-dir", default=DEFAULT_SAM3_CODE_DIR)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional, e.g. 0. Must be set before model import.")

    parser.add_argument("--categories", default=None, help="Comma-separated valid categories shown to the agent, e.g. 'thyroid nodule,thyroid gland'")
    parser.add_argument("--categories-file", default=None, help="Optional .txt/.json category list shown to the agent")

    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY") or "")
    parser.add_argument("--api-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.bitidea.cn/v1/chat/completions"))
    parser.add_argument("--api-model", default=os.environ.get("OPENAI_MODEL", "gemini-3-pro-preview-11-2025"))
    parser.add_argument("--api-timeout", type=int, default=180)
    parser.add_argument("--api-max-tokens", type=int, default=512)
    parser.add_argument("--api-image-max-side", type=int, default=768)
    parser.add_argument("--api-temperature", type=float, default=0.0)
    parser.add_argument("--api-fallback-on-error", action="store_true", default=True)
    parser.add_argument("--no-api-fallback-on-error", dest="api_fallback_on_error", action="store_false")
    parser.add_argument("--no-api", action="store_true", help="Skip API agent and use local fallback prompt extraction")

    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--max-area-ratio", type=float, default=1.0)
    parser.add_argument("--remove-overlap", action="store_true", help="Apply SAM3 overlap-removal helper if available")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    for p, label in [
        (args.image, "image"),
        (args.sam3_code_dir, "sam3 code dir"),
        (args.config_path, "config"),
        (args.checkpoint_path, "checkpoint"),
    ]:
        if not os.path.exists(os.path.expanduser(p)):
            raise FileNotFoundError(f"{label} not found: {p}")

    categories = parse_categories(args.categories, args.categories_file)
    print(f"[INFO] Agent category options: {len(categories)}")

    parsed, raw_agent = agent_parse_question(args.image, args.question, categories, args)
    print(f"[INFO] Agent status: {parsed.get('status')}")
    print(f"[INFO] Agent chosen category: {parsed.get('chosen_category_name')}")
    print(f"[INFO] SAM3 prompt: {parsed.get('sam3_prompt')}")

    model, device, load_info = setup_model(
        config_path=os.path.abspath(os.path.expanduser(args.config_path)),
        checkpoint_path=os.path.abspath(os.path.expanduser(args.checkpoint_path)),
        sam3_code_dir=os.path.abspath(os.path.expanduser(args.sam3_code_dir)),
    )

    serialized = sam3_predict_image(
        model=model,
        device=device,
        image_path=args.image,
        text_prompt=parsed["sam3_prompt"],
        remove_overlap=args.remove_overlap,
    )
    candidates = collect_candidates(serialized, parsed["sam3_prompt"], args.min_area, args.max_area_ratio)

    outputs = save_outputs(
        image_path=args.image,
        question=args.question,
        parsed=parsed,
        raw_agent_response=raw_agent,
        candidates=candidates,
        serialized=serialized,
        args=args,
        load_info=load_info,
    )

    print("\n[DONE] Single-image agent inference finished.")
    print(f"[DONE] Mask:       {outputs['mask']}")
    print(f"[DONE] Overlay:    {outputs['overlay']}")
    print(f"[DONE] Meta:       {outputs['meta']}")
    print(f"[DONE] Candidates: {outputs['candidates']}")
    print(f"[DONE] Run dir:    {outputs['run_dir']}")


if __name__ == "__main__":
    main()
