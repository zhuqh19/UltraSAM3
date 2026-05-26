
import os
import json
import cv2
import numpy as np
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime

# Paths
DATASET_ROOT = r'C:\Users\zhuqh\Desktop\sam3\datasets\乳腺超声数据集\11.乳腺超声(BUS_BRA)\BUSBRA'
IMAGES_DIR = os.path.join(DATASET_ROOT, 'Images')
MASKS_DIR = os.path.join(DATASET_ROOT, 'Masks')
CSV_PATH = os.path.join(DATASET_ROOT, 'bus_data.csv')
OUTPUT_DIR = r'C:\Users\zhuqh\Desktop\sam3\datasets\乳腺超声数据集\BUS_BRA_coco'

# Categories (Multi-label)
CATEGORIES = [
    {"supercategory": "breast lesion", "id": 0, "name": "breast lesion"},
    {"supercategory": "breast lesion", "id": 1, "name": "benign breast tumor"},
    {"supercategory": "breast lesion", "id": 2, "name": "malignant breast tumor"},
    {"supercategory": "breast lesion", "id": 3, "name": "breast tumor"},
]

GENERIC_LESION_ID = 0
BENIGN_ID = 1
MALIGNANT_ID = 2

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def binary_mask_to_polygon(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            contour = contour.flatten().tolist()
            if len(contour) > 4:
                polygons.append(contour)
    return polygons

def create_annotation_info(annotation_id, image_id, category_id, binary_mask):
    polygons = binary_mask_to_polygon(binary_mask)
    if not polygons:
        return None
    
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
        
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    bbox = [int(x_min), int(y_min), int(width), int(height)]
    area = int(np.sum(binary_mask > 0))
    
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": polygons,
        "area": area,
        "bbox": bbox,
        "iscrowd": 0,
    }
    return annotation

def process_split(df_split, split_name):
    split_dir = create_coco_structure(split_name)
    
    coco_output = {
        "info": {
            "description": "BUS_BRA dataset converted to COCO format",
            "url": "",
            "version": "1.0",
            "year": 2024,
            "contributor": "TraeAI",
            "date_created": datetime.now().strftime("%Y/%m/%d"),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": CATEGORIES,
    }
    
    annotation_id = 1
    
    print(f"Processing {split_name} split with {len(df_split)} images...")
    
    for i, row in df_split.iterrows():
        # Filename logic: ID + .png
        # Example: bus_0001-l -> bus_0001-l.png
        image_name = f"{row['ID']}.png"
        mask_name = f"mask_{row['ID'].replace('bus_', '')}.png"
        
        src_image_path = os.path.join(IMAGES_DIR, image_name)
        src_mask_path = os.path.join(MASKS_DIR, mask_name)
        dst_image_path = os.path.join(split_dir, image_name)
        
        # Check if files exist
        if not os.path.exists(src_image_path):
            print(f"Warning: Image not found {src_image_path}")
            continue
            
        # Copy image
        try:
            shutil.copy2(src_image_path, dst_image_path)
        except Exception as e:
            print(f"Warning: Failed to copy {src_image_path}: {e}")
            continue
            
        # Read image info
        img = read_image(src_image_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        image_id = len(coco_output["images"]) + 1
        
        # Classification from CSV
        pathology = row['Pathology'].lower()
        classification = "normal"
        if pathology == 'benign':
            classification = "benign"
        elif pathology == 'malignant':
            classification = "malignant"
            
        image_info = {
            "id": image_id,
            "file_name": image_name,
            "height": h,
            "width": w,
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "classification": classification
        }
        coco_output["images"].append(image_info)
        
        # Process Mask
        if os.path.exists(src_mask_path):
            mask = read_image(src_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                binary_mask = (mask > 0).astype(np.uint8)
                
                # Determine specific category ID
                specific_cat_id = None
                if classification == 'benign':
                    specific_cat_id = BENIGN_ID
                elif classification == 'malignant':
                    specific_cat_id = MALIGNANT_ID
                
                if specific_cat_id:
                    # 1. Specific
                    annotation = create_annotation_info(annotation_id, image_id, specific_cat_id, binary_mask)
                    if annotation:
                        coco_output["annotations"].append(annotation)
                        annotation_id += 1
                    
                    # 2. Generic
                    annotation = create_annotation_info(annotation_id, image_id, GENERIC_LESION_ID, binary_mask)
                    if annotation:
                        coco_output["annotations"].append(annotation)
                        annotation_id += 1
        else:
            print(f"Warning: Mask not found for {image_name} at {src_mask_path}")

    # Save JSON
    json_path = os.path.join(split_dir, '_annotations.coco.json')
    with open(json_path, 'w') as f:
        json.dump(coco_output, f, indent=4)
    print(f"Saved {split_name} annotations to {json_path}")

def main():
    # Read CSV
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"Found {len(df)} records in CSV.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
        
    # Split
    # Stratified by Pathology
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Pathology'])
    
    process_split(train_df, 'train')
    process_split(test_df, 'test') 
    
if __name__ == '__main__':
    main()
