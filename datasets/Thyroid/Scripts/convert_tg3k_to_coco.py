import os
import json
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from datetime import datetime
import glob

# Paths
DATASET_ROOT = r"C:\Users\zhuqh\Desktop\sam3\datasets\甲状腺超声数据集\38.甲状腺超声(TG3K)\tg3k"
IMAGES_DIR = os.path.join(DATASET_ROOT, "thyroid-image")
MASKS_DIR = os.path.join(DATASET_ROOT, "thyroid-mask")
OUTPUT_DIR = r"C:\Users\zhuqh\Desktop\sam3\datasets\甲状腺超声数据集\TG3K_coco"

# Categories
# No benign/malignant info readily available from filenames or simple JSON
# Using generic label
CATEGORIES = [
    {"id": 0, "name": "thyroid nodule"},
    {"id": 3, "name": "thyroid tumor"} # Generic specific label
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
    
    if not os.path.exists(IMAGES_DIR) or not os.path.exists(MASKS_DIR):
        print(f"Error: Missing directories in {DATASET_ROOT}")
        return []
        
    image_files = os.listdir(IMAGES_DIR)
    
    for img_f in image_files:
        if not img_f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
            
        # Match mask
        # 0000.jpg -> 0000.jpg
        mask_path = os.path.join(MASKS_DIR, img_f)
        
        if os.path.exists(mask_path):
            data_list.append({
                "image_path": os.path.join(IMAGES_DIR, img_f),
                "mask_path": mask_path,
                "filename": img_f
            })
        else:
            # Try searching? 
            # It seems names match exactly based on LS output
            pass
            
    return data_list

def process_split(data_list, split_name):
    split_dir = create_coco_structure(split_name)
    
    coco_output = {
        "info": {
            "description": f"TG3K Thyroid Dataset {split_name} Split",
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
            
        # Threshold
        # Mask analysis showed many values [0..255] likely due to JPG compression artifacts
        # But high values (>127) clearly define the ROI
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 20: # Filter noise
                continue
                
            segmentation = contour.flatten().tolist()
            x, y, w, h = cv2.boundingRect(contour)
            bbox = [x, y, w, h]
            area = cv2.contourArea(contour)
            
            # 1. Specific Annotation (Thyroid Tumor - ID 3)
            annotation = {
                "id": annotation_id,
                "image_id": image_id_counter,
                "category_id": 3,
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
    # Ignore existing train/val json, use random 8:2 split as requested
    try:
        train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"Split failed: {e}")
        return
    
    # 3. Process
    process_split(train_data, 'train')
    process_split(test_data, 'test')
    
    print("All done!")

if __name__ == "__main__":
    main()
