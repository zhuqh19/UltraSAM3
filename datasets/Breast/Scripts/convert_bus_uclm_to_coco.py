
import os
import json
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from datetime import datetime

# Paths
DATASET_ROOT = r'C:\Users\zhuqh\Desktop\sam3\datasets\乳腺超声数据集\10.乳腺超声(BUS_UCLM)\BUS-UCLM Breast ultrasound lesion segmentation dataset\BUS-UCLM'
IMAGES_DIR = os.path.join(DATASET_ROOT, 'images')
MASKS_DIR = os.path.join(DATASET_ROOT, 'masks')
OUTPUT_DIR = r'C:\Users\zhuqh\Desktop\sam3\datasets\乳腺超声数据集\BUS_UCLM_coco'

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

def process_split(image_files, split_name):
    split_dir = create_coco_structure(split_name)
    
    coco_output = {
        "info": {
            "description": "BUS_UCLM dataset converted to COCO format",
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
    
    print(f"Processing {split_name} split with {len(image_files)} images...")
    
    for img_file in image_files:
        src_image_path = os.path.join(IMAGES_DIR, img_file)
        dst_image_path = os.path.join(split_dir, img_file)
        
        # Copy image
        try:
            shutil.copy2(src_image_path, dst_image_path)
        except Exception as e:
            print(f"Warning: Failed to copy {src_image_path}: {e}")
            continue
            
        # Read image
        img = read_image(src_image_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        image_id = len(coco_output["images"]) + 1
        
        # Determine classification from mask for metadata (optional)
        classification = "normal"
        mask_path = os.path.join(MASKS_DIR, img_file)
        if os.path.exists(mask_path):
             mask_color = read_image(mask_path)
             if mask_color is not None:
                 if np.any(np.all(mask_color == [0, 0, 255], axis=-1)): # Red
                     classification = "malignant"
                 elif np.any(np.all(mask_color == [0, 255, 0], axis=-1)): # Green
                     classification = "benign"

        image_info = {
            "id": image_id,
            "file_name": img_file,
            "height": h,
            "width": w,
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "classification": classification
        }
        coco_output["images"].append(image_info)
        
        # Process Mask
        if os.path.exists(mask_path):
            mask_color = read_image(mask_path) # BGR
            if mask_color is not None:
                # Process Benign (Green: [0, 255, 0])
                # Note: OpenCV BGR -> Green is [0, 255, 0]
                lower_green = np.array([0, 250, 0])
                upper_green = np.array([10, 255, 10])
                mask_benign = cv2.inRange(mask_color, lower_green, upper_green)
                
                if np.sum(mask_benign) > 0:
                    # 1. Specific
                    annotation = create_annotation_info(annotation_id, image_id, BENIGN_ID, mask_benign)
                    if annotation:
                        coco_output["annotations"].append(annotation)
                        annotation_id += 1
                    # 2. Generic
                    annotation = create_annotation_info(annotation_id, image_id, GENERIC_LESION_ID, mask_benign)
                    if annotation:
                        coco_output["annotations"].append(annotation)
                        annotation_id += 1
                        
                # Process Malignant (Red: [0, 0, 255])
                # Note: OpenCV BGR -> Red is [0, 0, 255]
                lower_red = np.array([0, 0, 250])
                upper_red = np.array([10, 10, 255])
                mask_malignant = cv2.inRange(mask_color, lower_red, upper_red)
                
                if np.sum(mask_malignant) > 0:
                    # 1. Specific
                    annotation = create_annotation_info(annotation_id, image_id, MALIGNANT_ID, mask_malignant)
                    if annotation:
                        coco_output["annotations"].append(annotation)
                        annotation_id += 1
                    # 2. Generic
                    annotation = create_annotation_info(annotation_id, image_id, GENERIC_LESION_ID, mask_malignant)
                    if annotation:
                        coco_output["annotations"].append(annotation)
                        annotation_id += 1

    # Save JSON
    json_path = os.path.join(split_dir, '_annotations.coco.json')
    with open(json_path, 'w') as f:
        json.dump(coco_output, f, indent=4)
    print(f"Saved {split_name} annotations to {json_path}")

def main():
    all_images = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith('.png')]
    print(f"Found {len(all_images)} total images.")
    
    # Split
    train_files, test_files = train_test_split(all_images, test_size=0.2, random_state=42)
    
    process_split(train_files, 'train')
    process_split(test_files, 'test') 
    
if __name__ == '__main__':
    main()
