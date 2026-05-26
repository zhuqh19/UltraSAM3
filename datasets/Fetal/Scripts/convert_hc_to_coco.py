import os
import glob
import cv2
import numpy as np
import json
import shutil
from datetime import datetime
from sklearn.model_selection import train_test_split

# Config
DATASET_ROOT = r"C:\Users\zhuqh\Desktop\sam3\datasets\胎儿超声数据集\26.胎儿头围超声(HC)\1327317"
OUTPUT_DIR = os.path.join(DATASET_ROOT, "coco_format")

# Categories
# Task is Fetal Head Segmentation.
# Mask values are 0 (background) and 255 (foreground).
CATEGORIES = [
    {"id": 1, "name": "fetal head"}
]

def create_coco_structure(split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    return split_dir

def binary_mask_to_polygon(binary_mask):
    # Ensure binary 0-1
    binary_mask = (binary_mask > 127).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if contour.size >= 6:
            polygon = contour.flatten().tolist()
            polygons.append(polygon)
    return polygons

def get_bbox(binary_mask):
    binary_mask = (binary_mask > 127).astype(np.uint8)
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return [float(xmin), float(ymin), float(xmax - xmin + 1), float(ymax - ymin + 1)]

def process_split(file_pairs, split_name):
    split_dir = create_coco_structure(split_name)
    print(f"Processing {split_name} split with {len(file_pairs)} images...")
    
    coco_output = {
        "info": {
            "description": f"Fetal Head Ultrasound HC Dataset {split_name} Split",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": CATEGORIES
    }
    
    current_image_id = 1
    current_annotation_id = 1
    
    for img_path, mask_path in file_pairs:
        basename = os.path.basename(img_path)
        
        # Copy image
        dst_img_path = os.path.join(split_dir, basename)
        shutil.copy2(img_path, dst_img_path)
        
        # Read image for dims
        # Use cv2.imdecode for unicode paths
        try:
            img_data = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Error reading image {img_path}")
                continue
            height, width = img.shape[:2]
        except Exception as e:
            print(f"Exception reading image {img_path}: {e}")
            continue
            
        image_info = {
            "id": current_image_id,
            "file_name": basename,
            "width": int(width),
            "height": int(height)
        }
        coco_output['images'].append(image_info)
        
        # Process Mask
        try:
            mask_data = np.fromfile(mask_path, dtype=np.uint8)
            mask = cv2.imdecode(mask_data, cv2.IMREAD_UNCHANGED)
            if mask is None:
                print(f"Error reading mask {mask_path}")
                continue
            
            # Resize if needed (should match)
            if mask.shape[:2] != (height, width):
                 mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
            polygons = binary_mask_to_polygon(mask)
            bbox = get_bbox(mask)
            
            if polygons and bbox:
                area = float(np.sum(mask > 127))
                
                ann = {
                    "id": current_annotation_id,
                    "image_id": current_image_id,
                    "category_id": 1, # fetal head
                    "segmentation": polygons,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                coco_output['annotations'].append(ann)
                current_annotation_id += 1
                
        except Exception as e:
            print(f"Exception processing mask {mask_path}: {e}")
            
        current_image_id += 1
        
    # Save JSON
    json_path = os.path.join(split_dir, '_annotations.coco.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=4)
        
    print(f"Saved {split_name} annotations to {json_path}")
    print(f"Images: {len(coco_output['images'])}, Annotations: {len(coco_output['annotations'])}")

def get_pairs(directory):
    pairs = []
    # Images end with _HC.png or _2HC.png etc.
    # Masks end with _HC_Annotation.png or _2HC_Annotation.png etc.
    # Pattern: {NAME}.png -> {NAME}_Annotation.png
    
    # Get all png files
    all_pngs = glob.glob(os.path.join(directory, "*.png"))
    
    # Filter out annotations to find potential images
    image_candidates = [f for f in all_pngs if "_Annotation.png" not in f]
    
    for img_path in image_candidates:
        basename = os.path.basename(img_path)
        name_part = os.path.splitext(basename)[0]
        
        # Construct expected mask name
        mask_name = name_part + "_Annotation.png"
        mask_path = os.path.join(directory, mask_name)
        
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
        else:
            # Some files might not have annotation? Or naming convention differs?
            # From LS output: 000_HC.png -> 000_HC_Annotation.png
            # 010_2HC.png -> Likely 010_2HC_Annotation.png? 
            # Let's check if there are any orphans.
            # print(f"Warning: No mask found for {basename}")
            pass
            
    return pairs

def main():
    train_dir = os.path.join(DATASET_ROOT, "training_set")
    
    print("Scanning training set...")
    train_pairs = get_pairs(train_dir)
    print(f"Found {len(train_pairs)} pairs in training_set.")
    
    if not train_pairs:
        print("No data found!")
        return

    # Split training_set into 80% train and 20% valid (test)
    train_split, val_split = train_test_split(train_pairs, test_size=0.2, random_state=42)
    
    print(f"Splitting into Train: {len(train_split)}, Valid/Test: {len(val_split)}")

    process_split(train_split, "train")
    process_split(val_split, "valid")
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()
