#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-image + text-prompt inference for US-SAM3 checkpoints:

1. Create a temporary one-image COCO dataset whose category name is the text prompt.
2. Initialize the SAM3 Hydra trainer from the same config/checkpoint style as case study.
3. Run trainer.run_val() on that one-image dataset.
4. Read dumps/coco_predictions_segm.json.
5. Select the top-1 prediction by score.
6. Save binary mask and overlay image.

Normal usage only requires image + prompt:

python infer_sam3_single_image_case_top1.py \
  --image /path/to/image.png \
  --prompt "thyroid nodule"

Optional expert arguments are provided for switching checkpoint/config/code dir, but the
user-facing inference inputs remain image + text prompt.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import io
import json
import shutil
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


# ========================= Defaults =========================
DEFAULT_SAM3_CODE_DIR = "../code"
DEFAULT_SAM3_CONFIG = "../config/config.yaml"
DEFAULT_US_SAM3_CKPT = "/home/Data2/zhuquanhao/US-SAM3/US-SAM3_weight/US-SAM3.pt"
DEFAULT_OUTPUT_ROOT = "./sam3_single_image_outputs"

SCORE_THRESHOLD = 0.5


# ========================= Small utilities copied/adapted from case study =========================
def ensure_rgb_image(img_path: str, h: Optional[int] = None, w: Optional[int] = None) -> Image.Image:
    image = Image.open(img_path).convert("RGB")
    if h is not None and w is not None and image.size != (w, h):
        image = image.resize((w, h), Image.Resampling.BILINEAR)
    return image


def colorize_overlay(image_rgb: Image.Image, mask: np.ndarray, alpha: float = 0.45, color=(255, 0, 0)) -> Image.Image:
    arr = np.asarray(image_rgb.convert("RGB")).copy()
    if mask is None:
        return Image.fromarray(arr)
    mask_bool = (mask > 0)
    overlay = arr.copy()
    overlay[mask_bool] = (np.array(color) * alpha + overlay[mask_bool] * (1 - alpha)).astype(np.uint8)
    return Image.fromarray(overlay)


def decode_coco_pred_mask(ann: dict, h: int, w: int) -> Optional[np.ndarray]:
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


def load_top1_mask_from_coco_predictions(pred_file: str, coco_gt: COCO) -> Tuple[np.ndarray, Optional[dict], int]:
    """Decode top-1 prediction by score for the single image in coco_gt."""
    img_ids = coco_gt.getImgIds()
    if len(img_ids) != 1:
        raise ValueError(f"This script expects exactly one image, got {len(img_ids)}")

    img_id = int(img_ids[0])
    info = coco_gt.loadImgs(img_id)[0]
    h, w = int(info["height"]), int(info["width"])
    empty = np.zeros((h, w), dtype=np.uint8)

    if not os.path.exists(pred_file):
        return empty, None, 0

    with open(pred_file, "r") as f:
        preds = json.load(f)

    img_preds = [ann for ann in preds if int(ann.get("image_id", -1)) == img_id and "segmentation" in ann]
    img_preds = sorted(img_preds, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    if not img_preds:
        return empty, None, 0

    top1 = img_preds[0]
    mask = decode_coco_pred_mask(top1, h, w)
    if mask is None:
        mask = empty
    return mask, top1, len(img_preds)


# ========================= One-image COCO construction =========================
def make_single_image_coco_dataset(image_path: str, prompt: str, work_dir: Path) -> Dict[str, str]:
    """
    Build a minimal one-image COCO dataset.

    Important: in the case-study/eval path, SAM3 gets the text concept from COCO
    category names. Therefore we put the user's prompt as categories[0]['name'].

    A full-image dummy annotation is included to make dataloaders/evaluators that expect
    at least one annotation/category robust. The dummy GT is not used as the final output;
    only dumped predictions are decoded after run_val().
    """
    src = Path(image_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Image not found: {src}")

    image = Image.open(src).convert("RGB")
    w, h = image.size

    img_dir = work_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    file_name = src.name
    dst = img_dir / file_name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)

    # A legal full-image polygon. This is only a dummy annotation so COCO/dataloader
    # can expose the prompt category; inference output is read from prediction JSON.
    x2 = max(w - 1, 1)
    y2 = max(h - 1, 1)
    annotation = {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "segmentation": [[0, 0, x2, 0, x2, y2, 0, y2]],
        "area": float(max(w * h, 1)),
        "bbox": [0, 0, float(w), float(h)],
        "iscrowd": 0,
    }

    coco_dict = {
        "info": {"description": "single-image text-prompt inference dataset"},
        "licenses": [],
        "images": [{"id": 1, "file_name": file_name, "width": int(w), "height": int(h)}],
        "annotations": [annotation],
        "categories": [{"id": 1, "name": prompt, "supercategory": "object"}],
    }

    ann_file = work_dir / "_annotations.single_image.coco.json"
    with open(ann_file, "w") as f:
        json.dump(coco_dict, f)

    return {"img_folder": str(img_dir), "ann_file": str(ann_file), "file_name": file_name, "height": h, "width": w}


# ========================= SAM3 trainer path copied/adapted from case study =========================
def init_sam3_trainer(model_name: str, checkpoint_path: str, config_path: str, output_dir: str, sam3_code_dir: str):
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


def run_sam3_single_image_case_path(
    image_path: str,
    prompt: str,
    checkpoint_path: str,
    config_path: str,
    sam3_code_dir: str,
    output_root: str,
    model_name: str = "US-SAM3",
    debug: bool = False,
) -> Dict[str, str]:
    from hydra.utils import instantiate

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(image_path).stem
    run_dir = Path(output_root).expanduser().resolve() / f"{stem}_{ts}"
    work_dir = run_dir / "single_input"
    model_out = run_dir / "sam3_raw_outputs" / model_name
    current_output_dir = model_out / "single_image"
    for d in [work_dir, current_output_dir]:
        d.mkdir(parents=True, exist_ok=True)

    dataset_info = make_single_image_coco_dataset(image_path, prompt, work_dir)

    trainer = init_sam3_trainer(model_name, checkpoint_path, config_path, str(model_out), sam3_code_dir)

    trainer.data_conf.val.dataset.img_folder = dataset_info["img_folder"]
    trainer.data_conf.val.dataset.ann_file = dataset_info["ann_file"]
    if "pred_file_evaluators" in trainer.meters_conf.val.roboflow100.detection:
        trainer.meters_conf.val.roboflow100.detection.pred_file_evaluators[0].gt_path = dataset_info["ann_file"]
    trainer.meters_conf.val.roboflow100.detection.dump_dir = f"{current_output_dir}/dumps"
    trainer.logging_conf.log_dir = str(current_output_dir)

    trainer._setup_dataloaders()
    trainer.meters = instantiate(trainer.meters_conf, _convert_="all")

    f = io.StringIO()
    success = True
    with redirect_stdout(f):
        try:
            trainer.run_val()
        except Exception as e:
            success = False
            print(f"[ERROR] {model_name} failed: {repr(e)}")
            if debug:
                import traceback
                traceback.print_exc()
    stdout_text = f.getvalue()
    (run_dir / "run_val_stdout.txt").write_text(stdout_text)
    if debug:
        print(stdout_text)

    # Cleanup distributed state, matching the case-study comment/behavior.
    try:
        import torch
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        if debug:
            print(f"[WARN] distributed cleanup failed: {repr(e)}")

    del trainer

    if not success:
        raise RuntimeError(f"trainer.run_val() failed. See log: {run_dir / 'run_val_stdout.txt'}")

    pred_file = current_output_dir / "dumps" / "coco_predictions_segm.json"
    coco = COCO(dataset_info["ann_file"])
    mask, top1_ann, num_preds = load_top1_mask_from_coco_predictions(str(pred_file), coco)

    image = ensure_rgb_image(str(Path(dataset_info["img_folder"]) / dataset_info["file_name"]), dataset_info["height"], dataset_info["width"])

    mask_path = run_dir / f"{stem}_top1_mask.png"
    overlay_path = run_dir / f"{stem}_top1_overlay.png"
    meta_path = run_dir / f"{stem}_top1_meta.json"

    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(mask_path)
    colorize_overlay(image, mask).save(overlay_path)

    meta = {
        "image": str(Path(image_path).expanduser().resolve()),
        "prompt": prompt,
        "model_name": model_name,
        "checkpoint": checkpoint_path,
        "num_predictions_for_image": int(num_preds),
        "top1_score": None if top1_ann is None else float(top1_ann.get("score", 0.0)),
        "top1_category_id": None if top1_ann is None else top1_ann.get("category_id"),
        "top1_area_pixels": int(mask.sum()),
        "prediction_file": str(pred_file),
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "run_dir": str(run_dir),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    if top1_ann is None:
        print("[WARN] No predicted segmentation was found for this image. Saved an empty mask.")
        print(f"[WARN] Check raw prediction/log files under: {run_dir}")
    else:
        print(f"[INFO] Found {num_preds} predictions. Top1 score={meta['top1_score']:.6f}, area={meta['top1_area_pixels']} pixels")

    return {
        "run_dir": str(run_dir),
        "mask": str(mask_path),
        "overlay": str(overlay_path),
        "meta": str(meta_path),
        "pred_file": str(pred_file),
        "stdout": str(run_dir / "run_val_stdout.txt"),
    }


def parse_args():
    parser = argparse.ArgumentParser("Single-image SAM3 text-prompt inference via the case-study trainer path")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--prompt", required=True, help="Text concept prompt, e.g. 'thyroid nodule'")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Output root directory")
    parser.add_argument("--checkpoint", default=DEFAULT_US_SAM3_CKPT, help="SAM3-family checkpoint path; default is your US-SAM3 checkpoint")
    parser.add_argument("--sam3-code-dir", default=DEFAULT_SAM3_CODE_DIR, help="SAM3 code directory")
    parser.add_argument("--sam3-config", default=DEFAULT_SAM3_CONFIG, help="Internal Hydra config used by the case-study SAM3 trainer path")
    parser.add_argument("--model-name", default="US-SAM3", help="Name used only for output folders/logging")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional, e.g. 0. Set before SAM3 imports.")
    parser.add_argument("--debug", action="store_true", help="Print trainer stdout and traceback")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    for p, label in [
        (args.image, "image"),
        (args.checkpoint, "checkpoint"),
        (args.sam3_config, "sam3 config"),
        (args.sam3_code_dir, "sam3 code dir"),
    ]:
        if not os.path.exists(os.path.expanduser(p)):
            raise FileNotFoundError(f"{label} not found: {p}")

    outputs = run_sam3_single_image_case_path(
        image_path=args.image,
        prompt=args.prompt,
        checkpoint_path=os.path.abspath(os.path.expanduser(args.checkpoint)),
        config_path=os.path.abspath(os.path.expanduser(args.sam3_config)),
        sam3_code_dir=os.path.abspath(os.path.expanduser(args.sam3_code_dir)),
        output_root=args.output_root,
        model_name=args.model_name,
        debug=args.debug,
    )

    print("\n[DONE] Single-image inference finished.")
    print(f"[DONE] Mask:    {outputs['mask']}")
    print(f"[DONE] Overlay: {outputs['overlay']}")
    print(f"[DONE] Meta:    {outputs['meta']}")
    print(f"[DONE] Run dir: {outputs['run_dir']}")


if __name__ == "__main__":
    main()
