import os
import json
import cv2
import numpy as np
import shutil
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime

# Paths
DATASET_ROOT = r"C:\Users\zhuqh\Desktop\sam3\datasets\甲状腺超声数据集\39.甲状腺超声(Thyroid US Cineclip,36G多)"
HDF5_PATH = os.path.join(DATASET_ROOT, "dataset.hdf5")
METADATA_PATH = os.path.join(DATASET_ROOT, "metadata.csv")
OUTPUT_DIR = r"C:\Users\zhuqh\Desktop\sam3\datasets\甲状腺超声数据集\Thyroid_US_Cineclip_coco"

# Categories
# histopath_diagnosis: 0 (Benign), 1 (Malignant)
CATEGORIES = [
    {"id": 0, "name": "thyroid nodule"},
    {"id": 1, "name": "benign thyroid nodule"},
    {"id": 2, "name": "malignant thyroid nodule"}
]

def create_coco_structure(split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    return split_dir

def get_metadata_map():
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata not found at {METADATA_PATH}")
        return {}
        
    df = pd.read_csv(METADATA_PATH)
    meta_map = {}
    
    # annot_id maps to histopath_diagnosis
    for idx, row in df.iterrows():
        annot_id = str(row['annot_id']).strip()
        diagnosis = int(row['histopath_diagnosis'])
        meta_map[annot_id] = diagnosis
        
    return meta_map

def save_image(path, img):
    # cv2.imwrite fails with unicode paths on Windows
    # Use imencode + tofile
    is_success, im_buf = cv2.imencode(".png", img)
    if is_success:
        im_buf.tofile(path)
        return True
    return False

def process_split(indices, hdf5_path, meta_map, split_name):
    split_dir = create_coco_structure(split_name)
    print(f"Output directory for {split_name}: {split_dir}")
    
    coco_output = {
        "info": {
            "description": f"Thyroid US Cineclip Dataset {split_name} Split",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "User",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": CATEGORIES
    }
    
    annotation_id = 1
    image_id_counter = 1
    
    print(f"Processing {split_name} split with {len(indices)} frames...")
    
    # Open HDF5
    with h5py.File(hdf5_path, 'r') as f:
        images = f['image']
        masks = f['mask']
        annot_ids = f['annot_id']
        frame_nums = f['frame_num']
        
        for idx in indices:
            # Sort indices for efficient HDF5 access? 
            # Random access might be slow. But we are processing list.
            # Direct index access:
            
            try:
                # Read data
                img_data = images[idx]
                mask_data = masks[idx]
                annot_id_bytes = annot_ids[idx]
                frame_num = frame_nums[idx]
                
                # Decode ID
                annot_id = annot_id_bytes.decode('utf-8')
                
                # Get Diagnosis
                # 0: Benign -> ID 1
                # 1: Malignant -> ID 2
                diagnosis = meta_map.get(annot_id, 0) # Default to 0 if not found? Or skip?
                
                if diagnosis == 0:
                    category_id = 1
                else:
                    category_id = 2
                
                # Check if mask has content
                if np.max(mask_data) == 0:
                    continue
                    
                # Save Image
                # Filename: annotID_frameNum.png
                filename = f"{annot_id}_{frame_num}.png"
                filepath = os.path.join(split_dir, filename)
                
                # Image data is likely grayscale but check shape
                # Shape is (H, W) per previous check.
                # Convert to BGR for standard processing or keep gray
                if len(img_data.shape) == 2:
                    img_bgr = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
                else:
                    img_bgr = img_data
                
                if not save_image(filepath, img_bgr):
                    print(f"Failed to save {filename}")
                    continue
                    
                height, width = img_data.shape[:2]
                
                image_info = {
                    "id": image_id_counter,
                    "file_name": filename,
                    "width": width,
                    "height": height,
                    "date_captured": datetime.now().isoformat()
                }
                coco_output["images"].append(image_info)
                
                # Process Mask
                # Mask is 0-255? Or binary?
                # Previous check showed min 0 max 255.
                # Assume 255 is foreground.
                
                _, binary_mask = cv2.threshold(mask_data, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    if cv2.contourArea(contour) < 10:
                        continue
                        
                    segmentation = contour.flatten().tolist()
                    x, y, w, h = cv2.boundingRect(contour)
                    bbox = [x, y, w, h]
                    area = cv2.contourArea(contour)
                    
                    # 1. Specific Annotation
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id_counter,
                        "category_id": category_id,
                        "segmentation": [segmentation],
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0
                    }
                    coco_output["annotations"].append(annotation)
                    annotation_id += 1
                    
                    # 2. Generic Annotation
                    generic_annotation = annotation.copy()
                    generic_annotation["id"] = annotation_id
                    generic_annotation["category_id"] = 0
                    coco_output["annotations"].append(generic_annotation)
                    annotation_id += 1
                
                image_id_counter += 1
                
            except Exception as e:
                print(f"Error processing index {idx}: {e}")

    # Save JSON
    json_path = os.path.join(split_dir, '_annotations.coco.json')
    print(f"Saving COCO JSON to {json_path}...")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=4)
        
    print(f"Split {split_name} done. Images: {len(coco_output['images'])}, Annotations: {len(coco_output['annotations'])}")

def main():
    if not os.path.exists(HDF5_PATH):
        print("HDF5 file not found.")
        return
        
    meta_map = get_metadata_map()
    print(f"Loaded metadata for {len(meta_map)} subjects.")
    
    # We need to split by SUBJECT (annot_id), not by frame, to avoid leakage.
    # 1. Group indices by annot_id
    with h5py.File(HDF5_PATH, 'r') as f:
        annot_ids = f['annot_id'][:]
        # Convert to string list
        annot_ids_str = [x.decode('utf-8') for x in annot_ids]
        
    from collections import defaultdict
    subject_indices = defaultdict(list)
    for idx, aid in enumerate(annot_ids_str):
        subject_indices[aid].append(idx)
        
    unique_subjects = list(subject_indices.keys())
    print(f"Found {len(unique_subjects)} unique subjects in HDF5.")
    
    # 2. Split subjects
    try:
        train_subjects, test_subjects = train_test_split(unique_subjects, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"Split failed: {e}")
        return
        
    print(f"Train subjects: {len(train_subjects)}, Test subjects: {len(test_subjects)}")
    
    # 3. Flatten indices
    train_indices = []
    for sub in train_subjects:
        train_indices.extend(subject_indices[sub])
        
    test_indices = []
    for sub in test_subjects:
        test_indices.extend(subject_indices[sub])
        
    print(f"Train frames: {len(train_indices)}, Test frames: {len(test_indices)}")
    
    # 4. Process
    process_split(train_indices, HDF5_PATH, meta_map, 'train')
    process_split(test_indices, HDF5_PATH, meta_map, 'test')
    
    print("All done!")

if __name__ == "__main__":
    main()
