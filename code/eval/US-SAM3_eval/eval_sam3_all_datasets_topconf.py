import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import subprocess
import re
import json
import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
from datetime import datetime

# ===== Mask selection settings for overlap metrics =====
# The original script unions all predicted masks for each image.
# For text/concept-prompted segmentation, this can penalize models that output
# many candidate masks. Here we compute IoU/Dice after selecting high-confidence
# predictions.
# Options:
#   "top1"      : keep only the highest-score prediction per image. Recommended for most single-target ultrasound datasets.
#   "topk"      : keep the top K predictions per image. Useful for multi-instance tasks.
#   "threshold" : keep predictions whose score >= SCORE_THRESHOLD.
#   "topk_threshold": keep top K predictions after score filtering.
MASK_SELECTION_MODE = "top1"
TOP_K = 1
SCORE_THRESHOLD = 0.5


def _decode_pred_mask(ann, h, w):
    """Decode one COCO prediction annotation into a binary mask of shape (h, w)."""
    if "segmentation" not in ann:
        return None

    rle = ann["segmentation"]

    # Handle polygon/list-style RLE if necessary.
    if isinstance(rle, dict) and "counts" in rle and isinstance(rle["counts"], list):
        rle = maskUtils.frPyObjects(rle, h, w)

    m = maskUtils.decode(rle)

    # If the decoded mask is not at the original GT size, resize it back.
    if m.shape[0] != h or m.shape[1] != w:
        import cv2
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    # maskUtils.decode may return H x W x N for multiple RLEs.
    if m.ndim == 3:
        m = np.max(m, axis=2)

    return (m > 0).astype(np.uint8)


def _select_predictions_by_score(pred_anns, mode="top1", top_k=1, score_thr=0.5):
    """
    Select high-confidence predicted masks before computing image-level IoU/Dice.

    This is different from the previous union-all strategy. It is usually more
    suitable when one text prompt is expected to produce one target mask, because
    low-score false positives will not be merged into the final prediction.
    """
    pred_anns = [ann for ann in pred_anns if "segmentation" in ann]
    pred_anns = sorted(pred_anns, key=lambda x: float(x.get("score", 0.0)), reverse=True)

    if mode == "top1":
        return pred_anns[:1]

    if mode == "topk":
        return pred_anns[:top_k]

    if mode == "threshold":
        return [ann for ann in pred_anns if float(ann.get("score", 0.0)) >= score_thr]

    if mode == "topk_threshold":
        pred_anns = [ann for ann in pred_anns if float(ann.get("score", 0.0)) >= score_thr]
        return pred_anns[:top_k]

    if mode == "all":
        return pred_anns

    raise ValueError(f"Unknown MASK_SELECTION_MODE: {mode}")


def compute_iou_dice_from_coco(
    gt_file,
    pred_file,
    selection_mode=MASK_SELECTION_MODE,
    top_k=TOP_K,
    score_thr=SCORE_THRESHOLD,
):
    """
    Compute image-level mean IoU and Dice after selecting high-confidence masks.

    Compared with the previous implementation, this version does NOT union all
    predicted masks by default. It first selects masks by score, then unions the
    selected masks as the final prediction for each image.
    """
    if not os.path.exists(pred_file):
        return 0.0, 0.0

    coco_gt = COCO(gt_file)
    with open(pred_file, "r") as f:
        preds = json.load(f)

    if not preds:
        return 0.0, 0.0

    coco_dt = coco_gt.loadRes(preds)

    ious = []
    dices = []
    selected_counts = []
    raw_counts = []

    for img_id in coco_gt.getImgIds():
        img_info = coco_gt.loadImgs(img_id)[0]
        h, w = img_info["height"], img_info["width"]

        # GT: union all ground-truth instances into one foreground mask.
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        if len(ann_ids) > 0:
            gt_anns = coco_gt.loadAnns(ann_ids)
            for ann in gt_anns:
                gt_mask = np.maximum(gt_mask, coco_gt.annToMask(ann).astype(np.uint8))

        # Pred: select high-confidence masks first, then union selected masks.
        pred_ann_ids = coco_dt.getAnnIds(imgIds=img_id)
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        if len(pred_ann_ids) > 0:
            pred_anns_all = coco_dt.loadAnns(pred_ann_ids)
            raw_counts.append(len(pred_anns_all))
            pred_anns = _select_predictions_by_score(
                pred_anns_all,
                mode=selection_mode,
                top_k=top_k,
                score_thr=score_thr,
            )
            selected_counts.append(len(pred_anns))

            for ann in pred_anns:
                m = _decode_pred_mask(ann, h, w)
                if m is not None:
                    pred_mask = np.maximum(pred_mask, m)
        else:
            raw_counts.append(0)
            selected_counts.append(0)

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        pred_sum = pred_mask.sum()
        gt_sum = gt_mask.sum()

        if union == 0:
            # If both GT and prediction are empty, count as perfect.
            # If your datasets never contain empty GT, this branch will rarely be used.
            iou, dice = 1.0, 1.0
        else:
            iou = intersection / union
            dice = 0.0 if (pred_sum + gt_sum) == 0 else 2 * intersection / (pred_sum + gt_sum)

        ious.append(iou)
        dices.append(dice)

    return float(np.mean(ious)) if ious else 0.0, float(np.mean(dices)) if dices else 0.0

def run_evaluation(model_name, checkpoint_path, config_path, output_dir, test_datasets):
    print(f"\n{'='*50}")
    print(f"Evaluating {model_name}")
    print(f"{'='*50}")
    
    import sys
    sys.path.insert(0, "/home/Data2/zhuquanhao/sam3/code")
    import torch
    import io
    from contextlib import redirect_stdout
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from sam3.train.utils.train_utils import register_omegaconf_resolvers
    
    # Set env variables for single process run
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    
    config_dir = os.path.dirname(os.path.abspath(config_path))
    config_name = os.path.basename(config_path)
    
    try:
        register_omegaconf_resolvers()
    except Exception:
        pass
        
    initialize_config_dir(config_dir=config_dir, version_base="1.2")
    
    overrides = [
        f"trainer.mode=val",
        f"trainer.model.checkpoint_path={checkpoint_path}",
        f"launcher.experiment_log_dir={output_dir}",
        f"trainer.checkpoint.save_dir={output_dir}/checkpoints",
        "trainer.skip_first_val=False",
        "trainer.skip_saving_ckpts=True",
        f"trainer.meters.val.roboflow100.detection.iou_type=segm",
        f"+trainer.meters.val.roboflow100.detection.postprocessor.iou_type=segm",
        f"+trainer.meters.val.roboflow100.detection.postprocessor.convert_mask_to_rle=true",
        f"+trainer.meters.val.roboflow100.detection.postprocessor.use_original_sizes_mask=true"
    ]
    
    cfg = compose(config_name=config_name, overrides=overrides)
    cfg.launcher.num_nodes = 1
    cfg.launcher.gpus_per_node = 1
    
    # Instantiate trainer once. This loads the model and keeps it in memory!
    trainer = instantiate(cfg.trainer, _recursive_=False)
    
    results = {}
    
    for dataset_key, dataset_info in test_datasets.items():
        print(f"\n--- Testing on {dataset_key} ---")
        
        if not os.path.exists(dataset_info['ann_file']):
            print(f"Warning: Annotation file not found: {dataset_info['ann_file']}")
            results[dataset_key] = None
            continue

        current_output_dir = os.path.join(output_dir, dataset_key.replace("/", "_"))
        os.makedirs(current_output_dir, exist_ok=True)
        
        # Override data conf
        trainer.data_conf.val.dataset.img_folder = dataset_info['img_folder']
        trainer.data_conf.val.dataset.ann_file = dataset_info['ann_file']
        
        # Override meters conf
        if "pred_file_evaluators" in trainer.meters_conf.val.roboflow100.detection:
            trainer.meters_conf.val.roboflow100.detection.pred_file_evaluators[0].gt_path = dataset_info['ann_file']
        trainer.meters_conf.val.roboflow100.detection.dump_dir = f"{current_output_dir}/dumps"
        
        # Update logging dir
        trainer.logging_conf.log_dir = current_output_dir
        
        # Re-initialize dataloaders and meters
        trainer._setup_dataloaders()
        trainer.meters = instantiate(trainer.meters_conf, _convert_="all")
        
        # Run validation and capture stdout
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                trainer.run_val()
                success = True
            except Exception as e:
                import traceback
                print(f"Error evaluating on {dataset_key}: {e}")
                traceback.print_exc()
                success = False
                
        stdout_output = f.getvalue()
        
        # You can uncomment the next line if you want to see the captured output
        # print(stdout_output)
        
        if not success:
            results[dataset_key] = None
        else:
            # Parse mAP and AP from stdout
            metrics = parse_metrics(stdout_output)
            
            # Calculate IoU and Dice from dumped predictions
            pred_file = os.path.join(current_output_dir, "dumps", "coco_predictions_segm.json")
            mean_iou, mean_dice = compute_iou_dice_from_coco(dataset_info['ann_file'], pred_file)
            
            metrics['Mean IoU'] = mean_iou
            metrics['Mean Dice'] = mean_dice
            
            results[dataset_key] = metrics
            print(f"Results for {dataset_key}: {metrics}")
            
    return results

def parse_metrics(log_output):
    metrics = {}
    ap_50_95_match = re.search(r"Average Precision  \(AP\) @\[ IoU=0.50:0.95 \| area=   all \| maxDets=100 \] = (\d+\.\d+)", log_output)
    ap_50_match = re.search(r"Average Precision  \(AP\) @\[ IoU=0.50      \| area=   all \| maxDets=100 \] = (\d+\.\d+)", log_output)
    
    if ap_50_95_match:
        metrics['mAP (0.50:0.95)'] = float(ap_50_95_match.group(1))
    else:
        metrics['mAP (0.50:0.95)'] = 0.0
        
    if ap_50_match:
        metrics['AP (0.50)'] = float(ap_50_match.group(1))
    else:
        metrics['AP (0.50)'] = 0.0
        
    return metrics

def get_all_datasets():
    datasets_root = "/home/Data2/zhuquanhao/datasets"
    
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
        "Kidney": ["Ultrasound_Normal_Kidney_coco", "KidneyUS_coco"]
    }
    
    test_datasets = {}
    
    for organ, datasets in organs_datasets.items():
        for dataset_name in datasets:
            key = f"{organ}/{dataset_name}"
            
            organ_path_name = organ
            if organ == "Fetal": organ_path_name = "fetal"
            if organ == "Liver": organ_path_name = "liver"
            if organ == "Muscle": organ_path_name = "muscle"
            
            test_datasets[key] = {
                "name": dataset_name,
                "img_folder": f"{datasets_root}/{organ_path_name}/Datasets/{dataset_name}/test/",
                "ann_file": f"{datasets_root}/{organ_path_name}/Datasets/{dataset_name}/test/_annotations.coco.json"
            }
            
    return test_datasets

def main():
    # config_path = "/home/Data2/zhuquanhao/sam3/configs/all_datasets/config.yaml"
    config_path = "/home/Data2/zhuquanhao/sam3/configs/internal_datasets/config.yaml"
    finetuned_ckpt = "/home/Data2/zhuquanhao/sam3/logs/internal_datasets_exp_v2/checkpoints/checkpoint_2.pt"
    
    if not os.path.exists(finetuned_ckpt):
        print(f"Error: Checkpoint {finetuned_ckpt} not found.")
        return

    test_datasets = get_all_datasets()
    print(f"Found {len(test_datasets)} datasets to test.")
    print(f"IoU/Dice mask selection: mode={MASK_SELECTION_MODE}, top_k={TOP_K}, score_thr={SCORE_THRESHOLD}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/home/Data2/zhuquanhao/sam3/logs/eval_all_ft_{MASK_SELECTION_MODE}_{timestamp}"
    
    results_ft = run_evaluation("Fine-tuned Model", finetuned_ckpt, config_path, output_dir, test_datasets)
    
    print("\n" + "="*120)
    print(f"{'Dataset':<40} | {'mAP (0.50:0.95)':<20} | {'AP (0.50)':<20} | {'Mean IoU':<20} | {'Mean Dice':<20}")
    print("-" * 120)
    
    current_organ = ""
    
    # Also save summary to csv
    summary_file = os.path.join(output_dir, "summary.csv")
    with open(summary_file, "w") as f:
        f.write("Organ,Dataset,mAP(0.50:0.95),AP(0.50),Mean IoU,Mean Dice\n")
        for key in sorted(test_datasets.keys()):
            organ, dataset = key.split('/')
            
            if organ != current_organ:
                print(f"-- {organ} --")
                current_organ = organ
                
            res = results_ft.get(key)
            if res is None:
                continue
                
            def fmt(val): return f"{val:.4f}" if val is not None else "-"
            
            map_f = res.get('mAP (0.50:0.95)')
            ap50_f = res.get('AP (0.50)')
            iou_f = res.get('Mean IoU')
            dice_f = res.get('Mean Dice')
            
            print(f"{dataset:<40} | {fmt(map_f):<20} | {fmt(ap50_f):<20} | {fmt(iou_f):<20} | {fmt(dice_f):<20}")
            f.write(f"{organ},{dataset},{fmt(map_f)},{fmt(ap50_f)},{fmt(iou_f)},{fmt(dice_f)}\n")

    print(f"\nEvaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
