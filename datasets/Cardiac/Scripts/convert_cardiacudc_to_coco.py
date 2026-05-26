import os
import cv2
import json
import shutil
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm
import random
import re

# Dataset paths
DATASET_ROOT = r"\\?\C:\Users\zhuqh\Desktop\sam3\datasets\心脏超声数据集\15.心脏超声(CardiacUDC)\archive\cardiacUDC_dataset"
OUTPUT_DIR = r"\\?\C:\Users\zhuqh\Desktop\sam3\datasets\心脏超声数据集\CardiacUDC_coco"

# Folders containing labels (excluding Site_R_73 as it has no labels)
FOLDERS = ['label_all_frame', 'Site_G_100', 'Site_G_20', 'Site_G_29', 'Site_R_126', 'Site_R_52']

# COCO categories
# Assuming standard A4C labels:
# 1: Left Ventricle (LV)
# 2: Right Ventricle (RV)
# 3: Left Atrium (LA)
# 4: Right Atrium (RA)
CATEGORIES = [
    {"id": 1, "name": "left_ventricle", "supercategory": "heart"},
    {"id": 2, "name": "right_ventricle", "supercategory": "heart"},
    {"id": 3, "name": "left_atrium", "supercategory": "heart"},
    {"id": 4, "name": "right_atrium", "supercategory": "heart"},
]

def create_coco_structure():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)

def binary_mask_to_polygon(mask):
    """Convert binary mask to COCO polygon format."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) > 10:  # Filter small noise
            contour = contour.flatten().tolist()
            if len(contour) >= 6:  # Need at least 3 points
                polygons.append(contour)
    
    return polygons

def process_file_list(file_list, subset_name):
    output_subdir = os.path.join(OUTPUT_DIR, subset_name)
    os.makedirs(output_subdir, exist_ok=True)
    
    images = []
    annotations = []
    annotation_id = 1
    image_id = 1
    
    for img_path, label_path in tqdm(file_list, desc=f"Processing {subset_name}"):
        try:
            # Load NIfTI
            nii_img = nib.load(img_path)
            img_data = nii_img.get_fdata()
            
            nii_label = nib.load(label_path)
            label_data = nii_label.get_fdata()
            
            # Check dimensions match
            if img_data.shape != label_data.shape:
                print(f"Warning: Shape mismatch for {os.path.basename(img_path)}. Skipping.")
                continue
            
            # Iterate through frames (assuming last dimension is time/frames)
            num_frames = img_data.shape[-1]
            
            # Get filename stem for unique ID
            file_stem = os.path.splitext(os.path.splitext(os.path.basename(img_path))[0])[0]
            # Remove _image suffix if present
            if file_stem.endswith("_image"):
                file_stem = file_stem[:-6]
            
            for i in range(num_frames):
                frame_label = label_data[..., i]
                
                # Check if frame has any annotation
                if np.max(frame_label) == 0:
                    continue
                
                frame_img = img_data[..., i]
                
                # Rotate 90 degrees (common for medical NIfTI to image conversion)
                # Rotate -90 (270) degrees to fix orientation usually
                # Or try np.rot90(img, 1) -> 90 deg counter-clockwise
                # Let's use np.rot90(img, 1) which makes (W, H) -> (H, W) usually
                frame_img = np.rot90(frame_img)
                frame_label = np.rot90(frame_label)
                
                # Normalize image
                img_min = np.min(frame_img)
                img_max = np.max(frame_img)
                if img_max > img_min:
                    img_norm = ((frame_img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img_norm = np.zeros_like(frame_img, dtype=np.uint8)
                
                # Save image
                img_filename = f"{file_stem}_frame{i:03d}.jpg"
                img_output_path = os.path.join(output_subdir, img_filename)
                
                # Use cv2.imencode for unicode path support
                success, encoded_img = cv2.imencode(".jpg", img_norm)
                if success:
                    with open(img_output_path, "wb") as f:
                        f.write(encoded_img)
                
                # Add image info
                height, width = frame_img.shape
                images.append({
                    "id": image_id,
                    "file_name": img_filename,
                    "width": width,
                    "height": height
                })
                
                # Process masks for each category
                unique_labels = np.unique(frame_label)
                for label_val in unique_labels:
                    if label_val == 0:
                        continue
                    
                    # Create binary mask for this class
                    binary_mask = (frame_label == label_val).astype(np.uint8) * 255
                    polygons = binary_mask_to_polygon(binary_mask)
                    
                    for poly in polygons:
                        annotations.append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(label_val),
                            "segmentation": [poly],
                            "area": cv2.contourArea(np.array(poly).reshape(-1, 2).astype(np.float32)),
                            "bbox": cv2.boundingRect(np.array(poly).reshape(-1, 2).astype(np.float32)),
                            "iscrowd": 0
                        })
                        annotation_id += 1
                
                image_id += 1
                
        except Exception as e:
            print(f"Error processing {os.path.basename(img_path)}: {e}")
            continue

    # Save COCO JSON
    coco_output = {
        "info": {
            "description": f"CardiacUDC Dataset - {subset_name}",
            "year": 2024,
            "date_created": "2024-01-01"
        },
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES
    }
    
    json_path = os.path.join(output_subdir, "_annotations.coco.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=4)
        
    print(f"Saved {subset_name} annotations to {json_path}")

def get_patient_id(filename):
    # Try to extract patient ID from filename
    # Patterns: patient1-4, patient-1-4, normal-23-4
    # Remove _image.nii.gz suffix
    base = filename.replace("_image.nii.gz", "")
    # Remove trailing -number (view number)
    # Usually the last part after dash is view number?
    parts = base.split('-')
    if len(parts) > 1:
        return "-".join(parts[:-1])
    return base

def main():
    create_coco_structure()
    
    all_files = []
    
    print("Scanning folders...")
    for folder in FOLDERS:
        folder_path = os.path.join(DATASET_ROOT, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder} not found.")
            continue
            
        img_files = glob(os.path.join(folder_path, "*_image.nii.gz"))
        for img_path in img_files:
            # Construct expected label path
            label_path = img_path.replace("_image.nii.gz", "_label.nii.gz")
            if os.path.exists(label_path):
                all_files.append((img_path, label_path))
    
    print(f"Found {len(all_files)} total image/label pairs.")
    
    # Group by patient ID
    patient_files = {}
    for img_path, label_path in all_files:
        filename = os.path.basename(img_path)
        pid = get_patient_id(filename)
        if pid not in patient_files:
            patient_files[pid] = []
        patient_files[pid].append((img_path, label_path))
    
    patient_ids = list(patient_files.keys())
    print(f"Found {len(patient_ids)} unique patients.")
    
    # Shuffle and split 8:2
    random.seed(42)
    random.shuffle(patient_ids)
    
    split_idx = int(len(patient_ids) * 0.8)
    train_pids = patient_ids[:split_idx]
    test_pids = patient_ids[split_idx:]
    
    train_files = []
    for pid in train_pids:
        train_files.extend(patient_files[pid])
        
    test_files = []
    for pid in test_pids:
        test_files.extend(patient_files[pid])
        
    print(f"Train files: {len(train_files)} (from {len(train_pids)} patients)")
    print(f"Test files: {len(test_files)} (from {len(test_pids)} patients)")
    
    # Process
    if train_files:
        process_file_list(train_files, "train")
    if test_files:
        process_file_list(test_files, "test")
        
    print("Conversion completed!")

if __name__ == "__main__":
    main()
