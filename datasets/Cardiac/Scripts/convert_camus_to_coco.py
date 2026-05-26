import os
import cv2
import json
import shutil
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm
import random

# Dataset paths
DATASET_ROOT = r"\\?\C:\Users\zhuqh\Desktop\sam3\datasets\心脏超声数据集\14.心脏超声(CAMUS)\database_nifti"
OUTPUT_DIR = r"\\?\C:\Users\zhuqh\Desktop\sam3\datasets\心脏超声数据集\CAMUS_coco"

# COCO categories
# Standard CAMUS labels:
# 1: Left Ventricle Endocardium (LV_endo)
# 2: Left Ventricle Myocardium (LV_epi)
# 3: Left Atrium (LA)
CATEGORIES = [
    {"id": 1, "name": "left_ventricle_endocardium", "supercategory": "heart"},
    {"id": 2, "name": "left_ventricle_epicardium", "supercategory": "heart"},
    {"id": 3, "name": "left_atrium", "supercategory": "heart"},
]

def create_coco_structure():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)

def binary_mask_to_polygon(mask):
    """Convert binary mask to COCO polygon format."""
    # Ensure mask is binary (0/255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) > 10:  # Filter small noise
            contour = contour.flatten().tolist()
            if len(contour) >= 6:  # Need at least 3 points
                polygons.append(contour)
    
    return polygons

def process_patient_list(patient_list, subset_name):
    print(f"Processing {subset_name} ({len(patient_list)} patients)...")
    
    images = []
    annotations = []
    annotation_id = 1
    image_id = 1
    
    output_subdir = os.path.join(OUTPUT_DIR, subset_name)
    
    for patient_dir in tqdm(patient_list):
        patient_id = os.path.basename(patient_dir)
        
        # Define the 4 standard files per patient
        files_to_process = [
            (f"{patient_id}_2CH_ED.nii.gz", f"{patient_id}_2CH_ED_gt.nii.gz"),
            (f"{patient_id}_2CH_ES.nii.gz", f"{patient_id}_2CH_ES_gt.nii.gz"),
            (f"{patient_id}_4CH_ED.nii.gz", f"{patient_id}_4CH_ED_gt.nii.gz"),
            (f"{patient_id}_4CH_ES.nii.gz", f"{patient_id}_4CH_ES_gt.nii.gz"),
        ]
        
        for img_filename, mask_filename in files_to_process:
            img_path = os.path.join(patient_dir, img_filename)
            mask_path = os.path.join(patient_dir, mask_filename)
            
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                # Some patients might miss some views, skip gracefully
                continue
                
            try:
                # Load image
                nii_img = nib.load(img_path)
                img_data = nii_img.get_fdata()
                
                # Load mask
                nii_mask = nib.load(mask_path)
                mask_data = nii_mask.get_fdata()
                
                # Verify dimensions (should be 2D or 3D with 1 slice)
                if img_data.ndim == 3:
                    img_data = np.squeeze(img_data)
                if mask_data.ndim == 3:
                    mask_data = np.squeeze(mask_data)
                
                # Rotate image and mask if needed (NIfTI orientation can vary, but usually consistent within dataset)
                # For standard display, we often rotate 90 degrees or flip.
                # However, for training, consistency is key. We'll keep raw orientation unless it looks wrong.
                # Usually nibabel loads in RAS+ orientation.
                # Let's just rotate 90 degrees counter-clockwise to match typical medical view if needed.
                # Actually, let's keep it simple: transpose to (H, W) if it looks like (W, H).
                # But without visual confirmation, raw is safest. 
                # Wait, shape (549, 389) suggests (H, W) or (W, H). 
                # Standard conversion: usually rotate 90 deg. Let's apply a rotation to make it look upright if possible.
                # For now, raw data is fine for training.
                
                img_data = np.rot90(img_data) # Rotate 90 degrees counter-clockwise
                mask_data = np.rot90(mask_data)

                height, width = img_data.shape[:2]
                
                # Normalize image to 0-255 uint8
                img_min, img_max = np.min(img_data), np.max(img_data)
                if img_max > img_min:
                    img_norm = ((img_data - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img_norm = np.zeros_like(img_data, dtype=np.uint8)
                
                # Save image as JPG
                file_name = f"{patient_id}_{img_filename.replace('.nii.gz', '.jpg')}"
                dest_path = os.path.join(output_subdir, file_name)
                cv2.imencode(".jpg", img_norm)[1].tofile(dest_path)
                
                # Add image info
                image_info = {
                    "id": image_id,
                    "file_name": file_name,
                    "height": int(height),
                    "width": int(width)
                }
                images.append(image_info)
                
                # Process mask (multiclass)
                # 1: LV_endo, 2: LV_epi, 3: LA
                unique_labels = np.unique(mask_data)
                
                for label_val in unique_labels:
                    if label_val == 0:
                        continue
                        
                    category_id = int(label_val)
                    if category_id > 3: # Skip unexpected labels
                        continue
                        
                    binary_mask = (mask_data == label_val).astype(np.uint8) * 255
                    polygons = binary_mask_to_polygon(binary_mask)
                    
                    for poly in polygons:
                        annotation = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "segmentation": [poly],
                            "area": 0,
                            "bbox": [],
                            "iscrowd": 0
                        }
                        
                        # Calculate bbox and area
                        poly_np = np.array(poly).reshape((-1, 2))
                        x, y, w, h = cv2.boundingRect(poly_np.astype(np.int32))
                        annotation["bbox"] = [float(x), float(y), float(w), float(h)]
                        annotation["area"] = float(cv2.contourArea(poly_np.astype(np.int32)))
                        
                        annotations.append(annotation)
                        annotation_id += 1
                
                image_id += 1
                
            except Exception as e:
                print(f"Error processing {img_filename}: {e}")
                continue

    # Save COCO JSON
    coco_output = {
        "info": {
            "description": f"CAMUS Dataset - {subset_name}",
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

def main():
    create_coco_structure()
    
    # Get all patient directories
    patient_dirs = glob(os.path.join(DATASET_ROOT, "patient*"))
    patient_dirs = [d for d in patient_dirs if os.path.isdir(d)]
    
    print(f"Found {len(patient_dirs)} patients.")
    
    # Shuffle and split 8:2
    random.seed(42)
    random.shuffle(patient_dirs)
    
    split_idx = int(len(patient_dirs) * 0.8)
    train_patients = patient_dirs[:split_idx]
    test_patients = patient_dirs[split_idx:]
    
    print(f"Train patients: {len(train_patients)}")
    print(f"Test patients: {len(test_patients)}")
    
    # Process
    if train_patients:
        process_patient_list(train_patients, "train")
    if test_patients:
        process_patient_list(test_patients, "test")
        
    print("Conversion completed!")

if __name__ == "__main__":
    main()
