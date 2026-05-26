import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import base64
import gc
import json
import re
from io import BytesIO
from datetime import datetime

import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import pycocotools.mask as maskUtils
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

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


def clean_category_prompt(category_name: str) -> str:
    if category_name in CATEGORY_PROMPT_ALIASES:
        return CATEGORY_PROMPT_ALIASES[category_name]
    s = str(category_name).strip()
    s = s.replace("_", " ").replace("-", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s or "object"


def parse_target_categories(s: str, gt_category: str):
    if not s:
        return [gt_category]
    cats = []
    for x in re.split(r"[,;|]", s):
        x = x.strip()
        if x:
            cats.append(x)
    if gt_category not in cats:
        cats.insert(0, gt_category)
    return cats





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
                "ann_file": f"{datasets_root}/{organ_path_name}/Datasets/{dataset_name}/test/_annotations.coco.json",
            }
    return test_datasets


def build_organ_category_pools(datasets_root="/home/Data2/zhuquanhao/datasets"):
    """Build target category lists grouped by organ/domain from all COCO annotations."""
    test_datasets = get_all_datasets(datasets_root)
    pools = {}
    for dataset_key, info in test_datasets.items():
        organ = info["organ"]
        ann_file = info["ann_file"]
        if not os.path.exists(ann_file):
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
                "name": name,
                "meaning": meaning,
                "source_names": [],
                "source_datasets": [],
            })
            if name not in item["source_names"]:
                item["source_names"].append(name)
            if dataset_key not in item["source_datasets"]:
                item["source_datasets"].append(dataset_key)
    final = {}
    for organ, items in pools.items():
        options = []
        for i, (_, item) in enumerate(sorted(items.items(), key=lambda kv: kv[0]), start=1):
            item = dict(item)
            item["option_id"] = f"{organ}:{i}"
            item["organ"] = organ
            options.append(item)
        final[organ] = options
    return final


def parse_manual_target_options(s: str):
    options = []
    for i, x in enumerate(re.split(r"[,;|]", s or ""), start=1):
        x = x.strip()
        if not x:
            continue
        options.append({"option_id": str(i), "name": x, "meaning": clean_category_prompt(x), "source": "manual"})
    return options


def normalize_organ_name(name, valid_organs):
    if not name:
        return None
    low = str(name).strip().lower().replace(" ", "_")
    for organ in valid_organs:
        if organ.lower() == low or organ.lower().replace("_", " ") == low.replace("_", " "):
            return organ
    return None


def load_image_as_rgb(path):
    with Image.open(path) as im:
        im.load()
        return im.convert("RGB")


def pil_to_data_url(img, fmt="JPEG", quality=90):
    buf = BytesIO()
    img = img.convert("RGB")
    img.save(buf, format=fmt, quality=quality, optimize=True)
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def image_path_to_data_url(image_path, max_side=768):
    img = load_image_as_rgb(image_path)
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.BILINEAR)
    return pil_to_data_url(img, fmt="JPEG", quality=90)


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


def sam3_predict_image(model, device, image_path, text_prompt, remove_overlap=False):
    image_pil = load_image_as_rgb(image_path)
    batch_input, orig_h, orig_w = build_batch_for_sam3(image_pil, text_prompt, device)
    try:
        with torch.inference_mode():
            outputs = model(batch_input)
        pred_logits, pred_masks = extract_logits_masks(outputs, device)
        serialized = masks_to_serialized(pred_logits, pred_masks, orig_h, orig_w)
        if remove_overlap and remove_overlapping_masks is not None:
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


def select_top1(serialized, min_area=1, max_area_ratio=1.0):
    h, w = serialized["orig_img_h"], serialized["orig_img_w"]
    max_area = int(h * w * max_area_ratio) if max_area_ratio and max_area_ratio > 0 else None
    items = []
    for idx, rle in enumerate(serialized.get("pred_masks", [])):
        area = int(serialized.get("pred_areas", [0] * len(serialized.get("pred_masks", [])))[idx])
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        score = float(serialized.get("pred_scores", [0.0] * len(serialized.get("pred_masks", [])))[idx])
        box = serialized.get("pred_boxes", [[0, 0, 0, 0]] * len(serialized.get("pred_masks", [])))[idx]
        items.append({"idx": idx, "rle": rle, "score": score, "box": box, "area": area, "h": h, "w": w})
    items.sort(key=lambda x: x["score"], reverse=True)
    return items[0] if items else None


def decode_rle_to_mask(item):
    if not item:
        return None
    rle = dict(item["rle"])
    rle.setdefault("size", [item["h"], item["w"]])
    m = maskUtils.decode(rle)
    if m.ndim == 3:
        m = np.max(m, axis=2)
    return (m > 0).astype(np.uint8)


def overlay_mask(image_pil, mask, title, subtitle="", color=(255, 0, 0), alpha=0.45):
    img = image_pil.convert("RGB")
    out = img.copy()
    if mask is not None and mask.sum() > 0:
        overlay = Image.new("RGB", img.size, color)
        mask_img = Image.fromarray((mask * int(255 * alpha)).astype(np.uint8), mode="L")
        out = Image.composite(overlay, out, mask_img)
        draw = ImageDraw.Draw(out)
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            draw.rectangle([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], outline=color, width=3)
    canvas_h = out.height + 88
    canvas = Image.new("RGB", (out.width, canvas_h), (255, 255, 255))
    canvas.paste(out, (0, 88))
    draw = ImageDraw.Draw(canvas)
    try:
        font_b = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        font_b = None
        font = None
    draw.text((10, 8), title[:120], fill=(0, 0, 0), font=font_b)
    # wrap subtitle roughly
    lines = []
    text = subtitle or ""
    while len(text) > 0:
        lines.append(text[:130])
        text = text[130:]
        if len(lines) >= 3:
            break
    for i, line in enumerate(lines):
        draw.text((10, 34 + i * 17), line, fill=(30, 30, 30), font=font)
    return canvas


def make_side_by_side(left, right, summary_text=""):
    gap = 16
    w = left.width + right.width + gap
    h = max(left.height, right.height) + (70 if summary_text else 0)
    canvas = Image.new("RGB", (w, h), (245, 245, 245))
    if summary_text:
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 15)
        except Exception:
            font = None
        draw.text((10, 10), summary_text[:260], fill=(0, 0, 0), font=font)
        y0 = 70
    else:
        y0 = 0
    canvas.paste(left, (0, y0))
    canvas.paste(right, (left.width + gap, y0))
    return canvas


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


def generate_complex_instruction(image_path, gt_category, args):
    if args.complex_instruction:
        return args.complex_instruction.strip(), "provided", ""
    clean = clean_category_prompt(gt_category)
    system_text = (
        "You generate one complex but unambiguous user instruction for an ultrasound segmentation model. "
        "The instruction should describe the target using natural clinical/anatomical wording, not just repeat the raw label. "
        "Return ONLY valid JSON: {\"instruction\": \"...\"}."
    )
    content = [
        {"type": "text", "text": (
            f"Raw ground-truth target category: {gt_category!r}\n"
            f"Clean target meaning: {clean!r}\n"
            "Write one instruction under 35 words asking to segment this target in the image."
        )}
    ]
    if args.include_image_for_complex_generation:
        content.append({"type": "image_url", "image_url": {"url": image_path_to_data_url(image_path, args.api_image_max_side)}})
    messages = [{"role": "system", "content": system_text}, {"role": "user", "content": content}]
    try:
        raw = call_chat_api(messages, args, max_tokens=256)
        obj = extract_json_object(raw) or {}
        instr = str(obj.get("instruction", "")).strip()
        if not instr:
            instr = f"In this ultrasound image, identify and segment the anatomical structure corresponding to {clean}."
            return instr, "fallback_empty", raw
        return instr, "api", raw
    except Exception as e:
        instr = f"In this ultrasound image, identify and segment the anatomical structure corresponding to {clean}."
        return instr, f"fallback_error: {e}", ""



def detect_organ_with_agent(image_path, user_prompt, organ_pools, args):
    valid_organs = sorted(organ_pools.keys())
    if args.organ:
        organ = normalize_organ_name(args.organ, valid_organs)
        if organ:
            return {"organ": organ, "status": "provided", "reason": "organ provided by user", "raw_response": ""}
        # If user provided a nonstandard organ name, keep it as manual/fallback.
        return {"organ": args.organ, "status": "provided_unknown", "reason": "organ provided but not found in built pools", "raw_response": ""}

    system_text = (
        "You are a front-end routing agent for ultrasound segmentation. "
        "Given one ultrasound image and a user's segmentation prompt, choose the most likely organ/domain "
        "from the provided list. Do not segment the image. Return ONLY valid JSON: "
        "{\"organ\": \"...\", \"reason\": \"...\"}."
    )
    user_text = (
        f"User segmentation prompt: {user_prompt!r}\n\n"
        f"Valid organ/domain options: {json.dumps(valid_organs, ensure_ascii=False)}\n\n"
        "Choose exactly one option from the list."
    )
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_path_to_data_url(image_path, args.api_image_max_side)}},
        ]},
    ]
    try:
        raw = call_chat_api(messages, args, max_tokens=256)
        obj = extract_json_object(raw) or {}
        organ = normalize_organ_name(obj.get("organ", ""), valid_organs)
        if organ is None:
            # Fallback: simple keyword matching from prompt.
            p = user_prompt.lower()
            for o in valid_organs:
                if o.lower().replace("_", " ") in p or o.lower() in p:
                    organ = o
                    break
        if organ is None:
            organ = valid_organs[0] if valid_organs else "Unknown"
            status = "fallback_invalid"
        else:
            status = "api"
        return {"organ": organ, "status": status, "reason": str(obj.get("reason", "")), "raw_response": raw}
    except Exception as e:
        organ = valid_organs[0] if valid_organs else "Unknown"
        return {"organ": organ, "status": f"fallback_error: {e}", "reason": "API organ routing failed", "raw_response": ""}


def parse_instruction_with_agent(image_path, user_prompt, target_options, organ, args):
    categories_text = json.dumps([
        {
            "option_id": str(c.get("option_id", i + 1)),
            "name": c.get("name", "object"),
            "meaning": c.get("meaning", clean_category_prompt(c.get("name", "object"))),
            "source_names": c.get("source_names", []),
        }
        for i, c in enumerate(target_options)
    ], ensure_ascii=False, indent=2)
    system_text = (
        "You are a lightweight front-end agent for SAM3 ultrasound segmentation. "
        "You receive an ultrasound image, a user prompt, the selected organ/domain, and a fixed list of valid target categories "
        "restricted to that organ. Your job is NOT to segment masks. Your job is to choose exactly one target category "
        "from the list and rewrite the user prompt into a short SAM3-friendly segmentation prompt. "
        "Do not invent categories outside the list. Return ONLY valid JSON with keys: "
        "chosen_option_id, chosen_category, sam3_prompt, reason. Keep sam3_prompt under 14 words."
    )
    user_text = (
        f"Selected organ/domain: {organ}\n"
        f"User segmentation prompt: {user_prompt!r}\n\n"
        f"Valid target categories within this organ/domain:\n{categories_text}\n\n"
        "Choose the single best matching category. Then write a concise SAM3 prompt, preferably a noun phrase."
    )
    content = [{"type": "text", "text": user_text}]
    if args.agent_include_image:
        content.append({"type": "image_url", "image_url": {"url": image_path_to_data_url(image_path, args.api_image_max_side)}})
    messages = [{"role": "system", "content": system_text}, {"role": "user", "content": content}]
    try:
        raw = call_chat_api(messages, args, max_tokens=512)
        obj = extract_json_object(raw) or {}
        chosen_option_id = str(obj.get("chosen_option_id", "")).strip()
        chosen_name = str(obj.get("chosen_category", "")).strip()
        prompt = str(obj.get("sam3_prompt", "")).strip()
        reason = str(obj.get("reason", "")).strip()

        chosen = None
        if chosen_option_id:
            for c in target_options:
                if str(c.get("option_id")) == chosen_option_id:
                    chosen = c
                    break
        if chosen is None and chosen_name:
            chosen_clean = clean_category_prompt(chosen_name).lower()
            for c in target_options:
                if c.get("name", "").lower() == chosen_name.lower() or c.get("meaning", "").lower() == chosen_clean:
                    chosen = c
                    break
        if chosen is None:
            chosen = target_options[0] if target_options else {"option_id": "0", "name": "object", "meaning": "object"}
            status = "fallback_invalid_choice"
        else:
            status = "api"
        if not prompt:
            prompt = chosen.get("meaning", clean_category_prompt(chosen.get("name", "object")))
        return {
            "chosen_option_id": str(chosen.get("option_id", "")),
            "chosen_category": chosen.get("name", "object"),
            "chosen_meaning": chosen.get("meaning", clean_category_prompt(chosen.get("name", "object"))),
            "sam3_prompt": re.sub(r"\s+", " ", prompt).strip(),
            "reason": reason,
            "status": status,
            "raw_response": raw,
        }
    except Exception as e:
        chosen = target_options[0] if target_options else {"option_id": "0", "name": "object", "meaning": "object"}
        return {
            "chosen_option_id": str(chosen.get("option_id", "")),
            "chosen_category": chosen.get("name", "object"),
            "chosen_meaning": chosen.get("meaning", clean_category_prompt(chosen.get("name", "object"))),
            "sam3_prompt": chosen.get("meaning", clean_category_prompt(chosen.get("name", "object"))),
            "reason": f"fallback because API failed: {e}",
            "status": "fallback_error",
            "raw_response": "",
        }


def main():
    parser = argparse.ArgumentParser(description="Single-image visual comparison: direct prompt vs organ-aware front API agent parsed prompt for SAM3.")
    parser.add_argument("--image_path", required=True, help="Path to one input image.")
    parser.add_argument("--prompt", default="", help="User prompt / complex instruction to send directly to SAM3 in branch A.")
    parser.add_argument("--complex_instruction", default="", help="Alias for --prompt. If both are set, --complex_instruction is used.")
    parser.add_argument("--gt_category", default="", help="Optional ground-truth category for metadata only; not required for agent routing.")
    parser.add_argument("--organ", default="", help="Optional known organ/domain, e.g. Cardiac, Thyroid, Breast. If omitted, API first routes to an organ.")
    parser.add_argument("--target_categories", default="", help="Optional manual comma/semicolon-separated target categories. If provided, these override the organ pool.")
    parser.add_argument("--datasets_root", default="/home/Data2/zhuquanhao/datasets", help="Root used to build organ-level target category pools from COCO annotations.")
    parser.add_argument("--output_dir", default="", help="Output directory. Default: /home/Data2/zhuquanhao/sam3/logs/single_image_organ_agent_compare_<timestamp>")
    parser.add_argument("--config_path", default="/home/Data2/zhuquanhao/sam3/configs/internal_datasets/config.yaml")
    parser.add_argument("--checkpoint_path", default="/home/Data2/zhuquanhao/sam3/logs/internal_datasets_exp_v2/checkpoints/checkpoint_2.pt")
    parser.add_argument("--remove_overlap", action="store_true")
    parser.add_argument("--min_area", type=int, default=1)
    parser.add_argument("--max_area_ratio", type=float, default=1.0)

    # API defaults copied from the previous agent scripts.
    parser.add_argument("--api_key", default="sk-L250dA5cmSvKImoyZQv8LmjBcCXbhU2XGtCGlCD9MAeSTo9D")
    parser.add_argument("--api_url", default="https://api.bitidea.cn/v1/chat/completions")
    parser.add_argument("--api_model", default="gpt-4o-2024-11-20")
    parser.add_argument("--api_timeout", type=int, default=180)
    parser.add_argument("--api_max_tokens", type=int, default=1024)
    parser.add_argument("--api_image_max_side", type=int, default=768)
    parser.add_argument("--agent_include_image", action="store_true", default=True, help="Let the front agent see the image while parsing instruction. Default true.")
    parser.add_argument("--no_agent_image", dest="agent_include_image", action="store_false")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(args.image_path)
    if not args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"/home/Data2/zhuquanhao/sam3/logs/single_image_organ_agent_compare_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)

    user_prompt = (args.complex_instruction or args.prompt or "").strip()
    if not user_prompt:
        if args.gt_category:
            user_prompt = f"Please segment the {clean_category_prompt(args.gt_category)} in this ultrasound image."
        else:
            raise ValueError("Please provide --prompt/--complex_instruction, or provide --gt_category so a simple prompt can be created.")

    print("Image:", args.image_path)
    print("User prompt:", user_prompt)
    if args.gt_category:
        print("GT category metadata:", args.gt_category)

    organ_pools = build_organ_category_pools(args.datasets_root)
    manual_options = parse_manual_target_options(args.target_categories)

    if manual_options:
        organ_info = {"organ": args.organ or "Manual", "status": "manual_categories", "reason": "manual --target_categories provided", "raw_response": ""}
        target_options = manual_options
    else:
        organ_info = detect_organ_with_agent(args.image_path, user_prompt, organ_pools, args)
        organ = organ_info["organ"]
        target_options = organ_pools.get(organ, [])
        if not target_options:
            # Last-resort fallback: use gt_category or generic object.
            fallback = args.gt_category or "object"
            target_options = [{"option_id": "fallback:1", "name": fallback, "meaning": clean_category_prompt(fallback), "source": "fallback"}]
            organ_info["status"] = organ_info.get("status", "") + "+fallback_no_pool"

    print("\nOrgan routing status:", organ_info["status"])
    print("Chosen organ/domain:", organ_info["organ"])
    print("Organ routing reason:", organ_info.get("reason", ""))
    print(f"Target options shown to agent: {len(target_options)}")
    for opt in target_options[:20]:
        print(f"  - {opt.get('option_id')}: {opt.get('name')} -> {opt.get('meaning')}")
    if len(target_options) > 20:
        print(f"  ... {len(target_options) - 20} more")

    agent = parse_instruction_with_agent(args.image_path, user_prompt, target_options, organ_info["organ"], args)
    print("\nAgent parse status:", agent["status"])
    print("Agent chosen category:", agent["chosen_category"])
    print("Agent SAM3 prompt:", agent["sam3_prompt"])
    print("Agent reason:", agent.get("reason", ""))

    model, device = setup_model(args.config_path, args.checkpoint_path)

    # A: direct user prompt to SAM3.
    print("\nRunning SAM3 A: direct user prompt...")
    direct_ser = sam3_predict_image(model, device, args.image_path, user_prompt, remove_overlap=args.remove_overlap)
    direct_top = select_top1(direct_ser, min_area=args.min_area, max_area_ratio=args.max_area_ratio)
    direct_mask = decode_rle_to_mask(direct_top)

    # B: organ-aware front agent parsed short prompt to SAM3.
    print("Running SAM3 B: organ-aware agent parsed prompt...")
    agent_ser = sam3_predict_image(model, device, args.image_path, agent["sam3_prompt"], remove_overlap=args.remove_overlap)
    agent_top = select_top1(agent_ser, min_area=args.min_area, max_area_ratio=args.max_area_ratio)
    agent_mask = decode_rle_to_mask(agent_top)

    img = load_image_as_rgb(args.image_path)
    base = os.path.splitext(os.path.basename(args.image_path))[0]
    direct_title = "A. Direct prompt → SAM3 top1"
    direct_sub = f"prompt={user_prompt} | score={direct_top['score']:.4f}, area={direct_top['area']}" if direct_top else f"prompt={user_prompt} | no mask"
    agent_title = "B. Organ-aware front API agent → SAM3 top1"
    agent_sub = f"organ={organ_info['organ']} | chosen={agent['chosen_category']} | prompt={agent['sam3_prompt']} | score={agent_top['score']:.4f}, area={agent_top['area']}" if agent_top else f"organ={organ_info['organ']} | chosen={agent['chosen_category']} | prompt={agent['sam3_prompt']} | no mask"

    direct_vis = overlay_mask(img, direct_mask, direct_title, direct_sub, color=(255, 64, 64), alpha=0.45)
    agent_vis = overlay_mask(img, agent_mask, agent_title, agent_sub, color=(64, 180, 255), alpha=0.45)
    summary = f"Prompt: {user_prompt} | Routed organ: {organ_info['organ']} | GT category: {args.gt_category or 'N/A'}"
    combined = make_side_by_side(direct_vis, agent_vis, summary_text=summary)

    direct_path = os.path.join(args.output_dir, f"{base}_A_direct_prompt_mask.png")
    agent_path = os.path.join(args.output_dir, f"{base}_B_organ_agent_parsed_mask.png")
    combined_path = os.path.join(args.output_dir, f"{base}_comparison.png")
    direct_vis.save(direct_path)
    agent_vis.save(agent_path)
    combined.save(combined_path)

    meta = {
        "image_path": args.image_path,
        "user_prompt": user_prompt,
        "gt_category": args.gt_category,
        "organ_routing": organ_info,
        "target_options": target_options,
        "agent": agent,
        "direct": {
            "prompt": user_prompt,
            "score": None if not direct_top else direct_top["score"],
            "area": None if not direct_top else direct_top["area"],
            "box": None if not direct_top else direct_top["box"],
            "raw_masks": len(direct_ser.get("pred_masks", [])),
        },
        "agent_parsed": {
            "prompt": agent["sam3_prompt"],
            "chosen_category": agent["chosen_category"],
            "chosen_meaning": agent.get("chosen_meaning"),
            "score": None if not agent_top else agent_top["score"],
            "area": None if not agent_top else agent_top["area"],
            "box": None if not agent_top else agent_top["box"],
            "raw_masks": len(agent_ser.get("pred_masks", [])),
        },
        "outputs": {
            "direct_mask_image": direct_path,
            "agent_mask_image": agent_path,
            "comparison_image": combined_path,
        },
    }
    meta_path = os.path.join(args.output_dir, f"{base}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nSaved outputs:")
    print("  Direct mask image :", direct_path)
    print("  Agent mask image  :", agent_path)
    print("  Comparison image  :", combined_path)
    print("  Metadata          :", meta_path)


if __name__ == "__main__":
    main()
