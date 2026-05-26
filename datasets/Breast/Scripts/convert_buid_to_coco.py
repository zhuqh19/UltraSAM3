
import os
import json
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from datetime import datetime
import glob

# Paths
DATASET_ROOT = r'C:\Users\zhuqh\Desktop\sam3\datasets\乳腺超声数据集\8.乳腺超声(BUID)'
BENIGN_DIR = os.path.join(DATASET_ROOT, 'Benign')
MALIGNANT_DIR = os.path.join(DATASET_ROOT, 'Malignant')
OUTPUT_DIR = r'C:\Users\zhuqh\Desktop\sam3\datasets\乳腺超声数据集\BUID_coco'

# Categories (Using the same multi-label strategy)
CATEGORIES = [
    {"supercategory": "breast lesion", "id": 0, "name": "breast lesion"},
    {"supercategory": "breast lesion", "id": 1, "name": "benign breast tumor"},
    {"supercategory": "breast lesion", "id": 2, "name": "malignant breast tumor"},
    {"supercategory": "breast lesion", "id": 3, "name": "breast tumor"},
]

GENERIC_LESION_ID = 0

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

def parse_filename(filename):
    """
    Parse filename like '1 Benign Image.bmp', '1 Benign Mask.tif'
    Returns: (id, type, kind) e.g., ('1', 'Benign', 'Image')
    """
    parts = filename.replace('.', ' ').split()
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2] # id, type, kind (Image/Mask/Lesion)
    return None, None, None

def get_data_list(folder_path, tumor_type):
    """
    Scan folder and pair images with masks.
    tumor_type: 'Benign' or 'Malignant'
    """
    data_list = []
    # Find all original images (ending with Image.bmp)
    # Note: glob might not handle space well in some OS, using listdir
    all_files = os.listdir(folder_path)
    
    image_files = [f for f in all_files if 'Image.bmp' in f]
    
    for img_file in image_files:
        # Construct mask filename
        # Pattern: 'X Benign Image.bmp' -> 'X Benign Mask.tif'
        # Some might be jpg or png, but based on LS output they are consistent
        base_name = img_file.replace(' Image.bmp', '')
        mask_file = base_name + ' Mask.tif'
        
        if mask_file in all_files:
            data_list.append({
                'image_path': os.path.join(folder_path, img_file),
                'mask_path': os.path.join(folder_path, mask_file),
                'type': tumor_type,
                'id': base_name.split()[0]
            })
        else:
            print(f"Warning: Mask not found for {img_file}")
            
    return data_list

def process_split(data_list, split_name):
    split_dir = create_coco_structure(split_name)
    
    coco_output = {
        "info": {
            "description": "BUID dataset converted to COCO format",
            "url": "https://www.kaggle.com/datasets/ aryashah2k/breast-ultrasound-images-dataset",
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
    
    print(f"Processing {split_name} split with {len(data_list)} images...")
    
    for i, item in enumerate(data_list):
        # Create a unique filename for COCO to avoid conflict if we merge later
        # e.g., benign_1.bmp
        ext = os.path.splitext(item['image_path'])[1]
        new_filename = f"{item['type'].lower()}_{item['id']}{ext}"
        
        image_id = len(coco_output["images"]) + 1
        
        dst_image_path = os.path.join(split_dir, new_filename)
        
        # Copy image
        try:
            shutil.copy2(item['image_path'], dst_image_path)
        except Exception as e:
            print(f"Warning: Failed to copy {item['image_path']}: {e}")
            continue
            
        # Read image
        img = read_image(item['image_path'])
        if img is None:
            continue
        h, w = img.shape[:2]
        
        image_info = {
            "id": image_id,
            "file_name": new_filename,
            "height": h,
            "width": w,
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "classification": item['type'].lower()
        }
        coco_output["images"].append(image_info)
        
        # Process Mask
        mask = read_image(item['mask_path'], cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            binary_mask = (mask > 0).astype(np.uint8)
            
            # Determine category ID
            # Benign -> 1, Malignant -> 2
            specific_cat_id = 1 if item['type'] == 'Benign' else 2
            
            # 1. Specific Annotation
            annotation = create_annotation_info(annotation_id, image_id, specific_cat_id, binary_mask)
            if annotation:
                coco_output["annotations"].append(annotation)
                annotation_id += 1
                
            # 2. Generic Annotation (Multi-label)
            annotation_generic = create_annotation_info(annotation_id, image_id, GENERIC_LESION_ID, binary_mask)
            if annotation_generic:
                coco_output["annotations"].append(annotation_generic)
                annotation_id += 1

    # Save JSON
    json_path = os.path.join(split_dir, '_annotations.coco.json')
    with open(json_path, 'w') as f:
        json.dump(coco_output, f, indent=4)
    print(f"Saved {split_name} annotations to {json_path}")

def main():
    # 1. Gather all data
    benign_data = get_data_list(BENIGN_DIR, 'Benign')
    malignant_data = get_data_list(MALIGNANT_DIR, 'Malignant')
    
    all_data = benign_data + malignant_data
    print(f"Found {len(benign_data)} benign and {len(malignant_data)} malignant cases.")
    
    # 2. Split
    # Stratified split based on type
    labels = [item['type'] for item in all_data]
    train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42, stratify=labels)
    
    # 3. Process
    process_split(train_data, 'train')
    process_split(test_data, 'test') 
    
if __name__ == '__main__':
    main()
