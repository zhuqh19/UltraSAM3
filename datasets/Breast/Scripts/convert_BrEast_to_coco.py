
import os
import json
import cv2
import numpy as np
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime

# Paths
DATASET_ROOT = r'C:\Users\zhuqh\Desktop\sam3\datasets\乳腺超声数据集\7.乳腺超声(BrEast)'
IMAGES_DIR = os.path.join(DATASET_ROOT, 'BrEaST-Lesions_USG-images_and_masks')
EXCEL_PATH = os.path.join(DATASET_ROOT, 'BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx')
OUTPUT_DIR = r'C:\Users\zhuqh\Desktop\sam3\datasets\乳腺超声数据集\BrEast_coco'

# Categories
CATEGORIES = [
    {"supercategory": "breast lesion", "id": 0, "name": "breast lesion"},
    {"supercategory": "breast lesion", "id": 1, "name": "benign breast tumor"},
    {"supercategory": "breast lesion", "id": 2, "name": "malignant breast tumor"},
    {"supercategory": "breast lesion", "id": 3, "name": "breast tumor"},
]

CAT_NAME_TO_ID = {
    "benign": 1,
    "malignant": 2,
    "normal": None 
}

# Generic category ID for "breast lesion"
GENERIC_LESION_ID = 0

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_coco_structure(split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    return split_dir

def read_image(path, flags=cv2.IMREAD_COLOR):
    """Read image with non-ASCII path support"""
    try:
        # np.fromfile reads the file into a byte array, then cv2.imdecode decodes it
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
            # COCO polygon format: [x1, y1, x2, y2, ...]
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
            "description": "BrEast dataset converted to COCO format",
            "url": "https://github.com/psi-fil/BrEaST-Lesions-USG-dataset",
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
        image_name = row['Image_filename']
        mask_name = row['Mask_tumor_filename']
        classification = row['Classification']
        
        image_id = len(coco_output["images"]) + 1

        src_image_path = os.path.join(IMAGES_DIR, image_name)
        dst_image_path = os.path.join(split_dir, image_name)
        
        # Copy image
        if not os.path.exists(src_image_path):
            print(f"Warning: Image not found {src_image_path}")
            continue
            
        try:
            shutil.copy2(src_image_path, dst_image_path)
        except Exception as e:
            print(f"Warning: Failed to copy {src_image_path} to {dst_image_path}: {e}")
            continue
        
        # Read image to get dimensions
        img = read_image(src_image_path)
        if img is None:
            print(f"Warning: Could not read image {src_image_path}")
            continue
        h, w = img.shape[:2]
        
        image_info = {
            "id": image_id,
            "file_name": image_name,
            "height": h,
            "width": w,
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "classification": classification
        }
        coco_output["images"].append(image_info)
        
        # Process Mask if not normal
        if classification != 'normal' and pd.notna(mask_name):
            mask_path = os.path.join(IMAGES_DIR, mask_name)
            if os.path.exists(mask_path):
                mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    binary_mask = (mask > 0).astype(np.uint8)
                    
                    category_id = CAT_NAME_TO_ID.get(classification)
                    if category_id:
                        # 1. Add specific annotation (Benign/Malignant)
                        annotation = create_annotation_info(annotation_id, image_id, category_id, binary_mask)
                        if annotation:
                            coco_output["annotations"].append(annotation)
                            annotation_id += 1
                        
                        # 2. Add generic annotation (Breast Lesion) - Multi-label strategy
                        # This ensures the model learns "breast lesion" explicitly
                        annotation_generic = create_annotation_info(annotation_id, image_id, GENERIC_LESION_ID, binary_mask)
                        if annotation_generic:
                            coco_output["annotations"].append(annotation_generic)
                            annotation_id += 1
            else:
                print(f"Warning: Mask not found for {image_name} at {mask_path}")

    # Save JSON
    json_path = os.path.join(split_dir, '_annotations.coco.json')
    with open(json_path, 'w') as f:
        json.dump(coco_output, f, indent=4)
    print(f"Saved {split_name} annotations to {json_path}")

def main():
    # Read Excel
    try:
        df = pd.read_excel(EXCEL_PATH)
    except Exception as e:
        print(f"Error reading excel: {e}")
        return

    print(f"Total cases: {len(df)}")
    
    # Stratified split
    # We split based on Classification to ensure balanced classes
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Classification'])
    
    process_split(train_df, 'train')
    process_split(test_df, 'test') 
    
if __name__ == '__main__':
    main()
