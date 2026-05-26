
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from hydra import compose, initialize
from hydra.utils import instantiate
from sam3.train.utils.train_utils import register_omegaconf_resolvers
from sam3.train.data.collator import collate_fn_api
from sam3.visualization_utils import render_masklet_frame, COLORS

def setup_model(cfg, checkpoint_path=None):
    model = instantiate(cfg.trainer.model)
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        # Use the logic from model_builder directly or just torch.load
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]
        
        # Handle possible detector prefix mismatch if any (though we fixed model_builder)
        # We'll just load strictly
        msg = model.load_state_dict(ckpt, strict=False)
        print(f"Load status: {msg}")
    
    model.to("cuda")
    model.eval()
    return model

def unnormalize_image(img_tensor):
    # img_tensor: (C, H, W) normalized
    # Assume mean/std from config: [0.5, 0.5, 0.5]
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(img_tensor.device)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)

def process_batch(model, batch):
    with torch.no_grad():
        # Move batch to device
        # batch is usually a DataClass, we need to move its tensors
        # Assuming batch has .to() method or we manually move parts
        # The collator returns a Batch object defined in sam3.train.data.collator
        
        # For simplicity, let's assume batch.to("cuda") works if implemented, 
        # or we inspect the batch object.
        # Looking at previous code, model(batch) works.
        # But we need to ensure batch is on GPU.
        
        # Let's try to move known fields
        if hasattr(batch, "img_batch"):
            batch.img_batch = batch.img_batch.to("cuda")
        
        outputs = model(batch)
        # outputs is typically a dict or SAM3Output
        
        # Post-process outputs
        # We need to convert outputs to masks/boxes
        # The model output is raw logits/masks.
        
        # We can use the postprocessor from config: sam3.eval.postprocessors.PostProcessImage
        # But let's just inspect outputs structure first or use a simple decoding
        
        return outputs

def get_gt_masks(batch):
    # batch.find_targets contains the GT
    # It's a list of dicts with 'masks', 'boxes', etc.
    return batch.find_targets

def visualize_comparison(img, gt_target, orig_output, ft_output, out_path):
    # Prepare visualization data
    # gt_target: dict with 'masks' (tensor)
    # outputs: SAM3 output
    
    # We need to decode the outputs.
    # SAM3 output usually has 'pred_masks', 'pred_boxes', 'pred_logits'
    # We need to threshold and select based on scores.
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # 1. Ground Truth
    ax = axes[0]
    ax.set_title("Ground Truth")
    ax.imshow(img)
    if gt_target and 'masks' in gt_target:
        masks = gt_target['masks'].cpu().numpy() # (N, H, W)
        # Combine masks for visualization or iterate
        # Use render_masklet_frame logic or simple overlay
        overlay = img.copy()
        if len(masks) > 0:
            # GT format conversion for visualization utils if needed
            # Let's use a simple overlay
            for i, mask in enumerate(masks):
                color = COLORS[i % len(COLORS)]
                mask = mask.astype(bool)
                # Apply color
                for c in range(3):
                    overlay[..., c] = np.where(mask, 
                                             overlay[..., c] * 0.5 + color[c] * 255 * 0.5, 
                                             overlay[..., c])
        ax.imshow(overlay.astype(np.uint8))
    ax.axis('off')

    # Helper to process model output
    def show_pred(ax, output, title):
        ax.set_title(title)
        ax.imshow(img)
        
        # Output is likely raw. We need to filter by score.
        # Assuming output has 'pred_logits' and 'pred_masks'
        # pred_logits: (B, Q, 2) or (B, Q, C) - binary usually
        # pred_masks: (B, Q, H, W)
        
        # For batch size 1:
        logits = output['pred_logits'][0] # (Q, 2)
        masks = output['pred_masks'][0]   # (Q, H, W)
        
        # Softmax for probabilities
        probs = torch.softmax(logits, dim=-1)[:, 0] # Class 0 is foreground/object usually? Or check config.
        # Wait, sam3 output might be different. 
        # Config says: loss_ce weight 20. 
        # Let's assume binary classification: index 0 is fg?
        # Actually in DETR, last class is usually background.
        
        # Let's check config "loss_fns_find".
        # It uses IABCEMdetr.
        
        # Let's just assume we pick top K or threshold.
        scores, labels = probs.max(dim=-1) # This might be wrong if binary is (fg, bg)
        # If binary: (B, Q, 1) -> sigmoid
        
        # Let's look at simple thresholding.
        # Or better, use the postprocessor.
        
        # Simplified: Filter by score > 0.5 (sigmoid) if shape is (Q, 1) or softmax if (Q, 2)
        if logits.shape[-1] == 1:
            scores = logits.sigmoid().squeeze(-1)
        else:
            scores = torch.softmax(logits, dim=-1)[:, 0] # Assuming 0 is the class
            
        keep = scores > 0.4 # Threshold from config matcher is 0.4
        
        pred_masks = masks[keep]
        pred_scores = scores[keep]
        
        # Resize masks to image size
        # masks are usually low res?
        # Config: downsample: False in semantic_seg, but find queries might be different.
        
        # Interpolate masks
        pred_masks = torch.nn.functional.interpolate(
            pred_masks.unsqueeze(1), 
            size=(img.shape[0], img.shape[1]), 
            mode="bilinear", 
            align_corners=False
        ).squeeze(1)
        
        pred_masks = (pred_masks > 0.0).cpu().numpy() # Logit mask threshold 0
        
        overlay = img.copy()
        for i, mask in enumerate(pred_masks):
            color = COLORS[(i + 5) % len(COLORS)] # Offset color
            mask = mask.astype(bool)
            for c in range(3):
                overlay[..., c] = np.where(mask, 
                                         overlay[..., c] * 0.5 + color[c] * 255 * 0.5, 
                                         overlay[..., c])
            
            # Draw score
            # Find centroid
            ys, xs = np.where(mask)
            if len(xs) > 0:
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                cv2.putText(overlay, f"{pred_scores[i]:.2f}", (cx, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                           
        ax.imshow(overlay.astype(np.uint8))
        ax.axis('off')

    # 2. Original Model
    # We need to access the dictionary correctly. 
    # SAM3 output is a list of dicts? Or a dict of lists?
    # It returns a SAM3Output which is a dict.
    # Let's try to access 'find_stages' -> 'find_outputs' -> last layer
    
    # We will implement robust extraction in the main loop by inspecting the object.
    
    # Placeholder for now
    ax2 = axes[1]
    ax2.set_title("Original Model")
    ax2.axis('off')
    
    ax3 = axes[2]
    ax3.set_title("Fine-tuned Model")
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif hasattr(obj, '__dataclass_fields__'):
        from dataclasses import fields
        for field in fields(obj):
            val = getattr(obj, field.name)
            setattr(obj, field.name, move_to_device(val, device))
        return obj
    return obj

def main():
    # 1. Config
    register_omegaconf_resolvers()
    with initialize(version_base=None, config_path="sam3/train/configs"):
        cfg = compose(config_name="custom_tumor_final.yaml")
    
    # 2. Dataset
    val_dataset = instantiate(cfg.trainer.data.val.dataset)
    collate_fn = instantiate(cfg.trainer.data.val.collate_fn)
    
    # 3. Models
    print("Loading Original Model...")
    orig_model = setup_model(cfg, "/root/sam3/weight/sam3.pt")
    
    print("Loading Fine-tuned Model...")
    # Find latest checkpoint
    ckpt_dir = "/root/sam3/logs/tumor_exp/checkpoints"
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    # sort by number
    ckpts.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    latest_ckpt = os.path.join(ckpt_dir, ckpts[-1])
    print(f"Using latest checkpoint: {latest_ckpt}")
    ft_model = setup_model(cfg, latest_ckpt)
    
    # 4. Output Dir
    out_dir = "/root/sam3/visualization_results"
    os.makedirs(out_dir, exist_ok=True)
    
    # 5. Iterate
    print(f"Visualizing {len(val_dataset)} images...")
    
    # Use a simple loader
    loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=1, 
        collate_fn=collate_fn, 
        num_workers=0
    )
    
    for i, batch_dict in enumerate(loader):
        print(f"Processing image {i+1}/{len(val_dataset)}...")
        
        # Extract batch
        # The key is 'roboflow100' based on config
        key = list(batch_dict.keys())[0]
        batch = batch_dict[key]
        
        # Move to GPU
        batch = move_to_device(batch, "cuda")
        
        # Run Inference
        with torch.no_grad():
            orig_out = orig_model(batch)
            ft_out = ft_model(batch)
            
        # Extract Image
        img = unnormalize_image(batch.img_batch[0])
        
        # Extract GT
        gt_target = batch.find_targets[0] if batch.find_targets else None
        
        # Visualization
        # We need to extract the actual prediction from the output structure
        # SAM3 output is complex.
        # Usually: out['find_stages'][-1] is the final output
        
        def extract_pred(out):
            # Try to find the final output
            if isinstance(out, list):
                # Maybe return from model is a tuple/list
                out = out[0]
                
            if 'find_stages' in out:
                return out['find_stages'][-1]
            return out # Fallback
            
        orig_pred = extract_pred(orig_out)
        ft_pred = extract_pred(ft_out)
        
        # Custom visualization
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # GT
        axes[0].set_title("Ground Truth")
        axes[0].imshow(img)
        if gt_target:
            # GT masks are in 'segments' key in BatchedFindTarget
            masks = None
            if hasattr(gt_target, 'segments'): 
                masks = gt_target.segments
            elif isinstance(gt_target, dict) and 'segments' in gt_target:
                masks = gt_target['segments']
                
            if masks is not None:
                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()
                elif isinstance(masks, list):
                    # Should be tensor if collated, but handle list just in case
                    if len(masks) > 0 and isinstance(masks[0], torch.Tensor):
                        masks = torch.stack(masks).cpu().numpy()
                    else:
                        masks = np.array(masks)
                
                # Check if masks is empty (0-d or 0 length)
                if masks.size > 0:
                    overlay = img.copy()
                    # masks shape: (N, H, W)
                    for m_idx, mask in enumerate(masks):
                        color = COLORS[m_idx % len(COLORS)]
                        mask = mask > 0
                        for c in range(3):
                            overlay[..., c] = np.where(mask, 
                                                     overlay[..., c] * 0.5 + color[c] * 255 * 0.5, 
                                                     overlay[..., c])
                    axes[0].imshow(overlay)
                else:
                     print("GT segments found but empty.")
            else:
                print("No segments found in GT target.")
        else:
            print("No GT target found.")
        axes[0].axis('off')
        
        # Model Preds
        for ax, pred, title in zip(axes[1:], [orig_pred, ft_pred], ["Original Model", "Fine-tuned Model"]):
            ax.set_title(title)
            ax.imshow(img)
            
            # Parse pred
            # pred is a dict with 'pred_logits', 'pred_masks'
            if 'pred_logits' in pred and 'pred_masks' in pred:
                logits = pred['pred_logits'][0] # (Q, C)
                masks = pred['pred_masks'][0]   # (Q, H, W)
                
                # Score
                # Assuming binary classification for tumor (fg/bg)
                # If output dim is 1: sigmoid. If 2: softmax.
                if logits.shape[-1] == 1:
                    scores = logits.sigmoid().squeeze(-1)
                else:
                    scores = torch.softmax(logits, dim=-1)[:, 0] # Index 0 for FG? Need to verify.
                    # In many DETR-like, index 0 is class, index 1 is 'no object' (background)
                    # Or last index is background.
                    # Given 'tumor' dataset has 1 class.
                    # Model config has cost_class: 2.0.
                    # Let's try taking max of non-background.
                
                # Heuristic: use max score for now
                if logits.shape[-1] > 1:
                    # Assuming 0 is the class we want (tumor)
                    scores = torch.softmax(logits, dim=-1)[:, 0]
                else:
                    scores = logits.sigmoid().flatten()
                
                # Threshold
                keep = scores > 0.4
                
                p_masks = masks[keep]
                p_scores = scores[keep]
                
                if len(p_masks) > 0:
                    # Upsample masks
                    p_masks = torch.nn.functional.interpolate(
                        p_masks.unsqueeze(1),
                        size=(img.shape[0], img.shape[1]),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(1)
                    p_masks = (p_masks > 0.0).cpu().numpy()
                    
                    overlay = img.copy()
                    for m_idx, mask in enumerate(p_masks):
                        color = COLORS[(m_idx + 2) % len(COLORS)]
                        mask = mask > 0
                        for c in range(3):
                            overlay[..., c] = np.where(mask, 
                                                     overlay[..., c] * 0.5 + color[c] * 255 * 0.5, 
                                                     overlay[..., c])
                        
                        # Draw score
                        ys, xs = np.where(mask)
                        if len(xs) > 0:
                            cx, cy = int(np.mean(xs)), int(np.mean(ys))
                            cv2.putText(overlay, f"{p_scores[m_idx]:.2f}", (cx, cy), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    ax.imshow(overlay)
                else:
                    ax.text(0.5, 0.5, "No Detection", transform=ax.transAxes, 
                           ha="center", color="red", fontsize=12)
            
            ax.axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"comparison_{i:03d}.png"))
        plt.close()
        
    print(f"Done! Results saved to {out_dir}")

if __name__ == "__main__":
    main()
