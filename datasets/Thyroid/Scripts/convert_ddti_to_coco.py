import os
import json
import cv2
import numpy as np
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime

# Paths
DATASET_ROOT = r"C:\Users\zhuqh\Desktop\sam3\datasets\甲状腺超声数据集\17.甲状腺超声(DDTI)\DDTI dataset\DDTI\1_or_data"
IMAGES_DIR = os.path.join(DATASET_ROOT, "image")
MASKS_DIR = os.path.join(DATASET_ROOT, "mask")
CATEGORY_CSV = os.path.join(DATASET_ROOT, "category.csv")
OUTPUT_DIR = r"C:\Users\zhuqh\Desktop\sam3\datasets\甲状腺超声数据集\DDTI_coco"

# Categories
# Based on category.csv: 0 and 1
# Assuming:
# 0: Benign
# 1: Malignant
# Need to confirm? Usually in medical datasets 0/1 often maps to Benign/Malignant.
# Let's use generic names if unsure, but usually:
# 0 -> Benign Thyroid Nodule
# 1 -> Malignant Thyroid Nodule
# Also add generic "thyroid nodule"

CATEGORIES = [
    {"id": 0, "name": "thyroid nodule"},
    {"id": 1, "name": "benign thyroid nodule"},
    {"id": 2, "name": "malignant thyroid nodule"}
]

def create_coco_structure(split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    return split_dir

def read_image(path, flags=cv2.IMREAD_COLOR):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    except Exception as e:
        print(f"Error reading image {path}: {e}")
        return None

def get_data_list():
    data_list = []
    
    if not os.path.exists(CATEGORY_CSV):
        print(f"Error: Category CSV not found at {CATEGORY_CSV}")
        return []
        
    try:
        df = pd.read_csv(CATEGORY_CSV)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []
        
    # Columns: ID, CATE
    # ID is filename like "1.PNG"
    # CATE is 0 or 1
    
    for index, row in df.iterrows():
        filename = str(row['ID']).strip()
        category_code = int(row['CATE'])
        
        image_path = os.path.join(IMAGES_DIR, filename)
        mask_path = os.path.join(MASKS_DIR, filename)
        
        if os.path.exists(image_path) and os.path.exists(mask_path):
            data_list.append({
                "image_path": image_path,
                "mask_path": mask_path,
                "filename": filename,
                "category_code": category_code
            })
        else:
            # Check for missing files
            if not os.path.exists(image_path):
                print(f"Warning: Image not found {image_path}")
            if not os.path.exists(mask_path):
                print(f"Warning: Mask not found {mask_path}")
                
    return data_list

def process_split(data_list, split_name):
    split_dir = create_coco_structure(split_name)
    
    coco_output = {
        "info": {
            "description": f"DDTI Thyroid Dataset {split_name} Split",
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
    
    print(f"Processing {split_name} split with {len(data_list)} images...")
    
    for item in data_list:
        # Copy image
        dst_image_path = os.path.join(split_dir, item['filename'])
        try:
            shutil.copy2(item['image_path'], dst_image_path)
        except Exception as e:
            print(f"Failed to copy {item['image_path']}: {e}")
            continue
            
        img = read_image(item['image_path'])
        if img is None:
            continue
        height, width = img.shape[:2]
        
        image_info = {
            "id": image_id_counter,
            "file_name": item['filename'],
            "width": width,
            "height": height,
            "date_captured": datetime.now().isoformat()
        }
        coco_output["images"].append(image_info)
        
        # Process Mask
        mask = read_image(item['mask_path'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            image_id_counter += 1
            continue
            
        # Threshold (Masks are 0/255)
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Determine Category ID
        # CSV: 0 -> Benign (ID 1), 1 -> Malignant (ID 2)
        # Note: Check logic. Usually 0 is Benign, 1 is Malignant in binary classification datasets.
        # We will assume:
        # Code 0 -> Benign -> COCO ID 1
        # Code 1 -> Malignant -> COCO ID 2
        
        if item['category_code'] == 0:
            specific_cat_id = 1
        else:
            specific_cat_id = 2
            
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 20: # Filter noise
                continue
                
            segmentation = contour.flatten().tolist()
            x, y, w, h = cv2.boundingRect(contour)
            bbox = [x, y, w, h]
            area = cv2.contourArea(contour)
            
            # 1. Specific Annotation
            annotation = {
                "id": annotation_id,
                "image_id": image_id_counter,
                "category_id": specific_cat_id,
                "segmentation": [segmentation],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            }
            coco_output["annotations"].append(annotation)
            annotation_id += 1
            
            # 2. Generic Annotation (Thyroid Nodule - ID 0)
            generic_annotation = annotation.copy()
            generic_annotation["id"] = annotation_id
            generic_annotation["category_id"] = 0
            coco_output["annotations"].append(generic_annotation)
            annotation_id += 1
        
        image_id_counter += 1

    # Save JSON
    json_path = os.path.join(split_dir, '_annotations.coco.json')
    print(f"Saving COCO JSON to {json_path}...")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=4)
        
    print(f"Split {split_name} done. Images: {len(coco_output['images'])}, Annotations: {len(coco_output['annotations'])}")

def main():
    # 1. Gather data
    all_data = get_data_list()
    print(f"Found {len(all_data)} valid image-mask pairs.")
    
    if not all_data:
        print("No data found!")
        return

    # 2. Split data (8:2)
    # Stratify by category code to ensure balanced split
    labels = [item['category_code'] for item in all_data]
    
    try:
        train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42, stratify=labels)
    except Exception as e:
        print(f"Stratified split failed: {e}. Falling back to random split.")
        train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
    
    # 3. Process
    process_split(train_data, 'train')
    process_split(test_data, 'test')
    
    print("All done!")

if __name__ == "__main__":
    main()
