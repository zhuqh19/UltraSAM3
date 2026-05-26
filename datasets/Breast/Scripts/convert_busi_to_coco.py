import os
import json
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from datetime import datetime
import glob

# Paths
DATASET_ROOT = r'C:\Users\zhuqh\Desktop\sam3\datasets\乳腺超声数据集\13.乳腺超声(BUSI)\Dataset_BUSI_with_GT'
OUTPUT_DIR = r'C:\Users\zhuqh\Desktop\sam3\datasets\乳腺超声数据集\BUSI_coco'

# Categories
CATEGORIES = [
    {"id": 0, "name": "breast lesion"},
    {"id": 1, "name": "benign breast tumor"},
    {"id": 2, "name": "malignant breast tumor"}
]

GENERIC_LESION_ID = 0

def read_image(path, flags=cv2.IMREAD_COLOR):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    except Exception as e:
        print(f"Error reading image {path}: {e}")
        return None

def get_data_list():
    data_list = []
    
    # Iterate over the three subdirectories
    for subdir in ['benign', 'malignant', 'normal']:
        dir_path = os.path.join(DATASET_ROOT, subdir)
        if not os.path.exists(dir_path):
            print(f"Warning: Directory not found: {dir_path}")
            continue
            
        print(f"Scanning {subdir}...")
        
        # Group files by base name
        # Image: "name.png"
        # Mask: "name_mask.png", "name_mask_1.png"
        
        files = os.listdir(dir_path)
        image_files = {} # name -> filename
        mask_files = {}  # name -> [filenames]
        
        for f in files:
            if not f.lower().endswith('.png'):
                continue
                
            name_no_ext = os.path.splitext(f)[0]
            
            # Check if it is a mask
            if '_mask' in name_no_ext:
                # Extract base name
                # benign (1)_mask -> benign (1)
                # benign (1)_mask_1 -> benign (1)
                
                parts = name_no_ext.split('_mask')
                base_name = parts[0]
                
                if base_name not in mask_files:
                    mask_files[base_name] = []
                mask_files[base_name].append(os.path.join(dir_path, f))
            else:
                # It is an image
                image_files[name_no_ext] = os.path.join(dir_path, f)
        
        # Match images with masks
        for base_name, img_path in image_files.items():
            current_masks = mask_files.get(base_name, [])
            
            # For normal images, masks might be empty or exist but be empty images
            # We treat 'normal' as having no lesions regardless of mask files (verified empty)
            # For benign/malignant, we need the masks.
            
            data_list.append({
                'image_path': img_path,
                'mask_paths': current_masks,
                'type': subdir, # benign, malignant, normal
                'filename': os.path.basename(img_path)
            })
            
    return data_list

def create_coco_structure(split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    return split_dir

def process_split(data_list, split_name):
    split_dir = create_coco_structure(split_name)
    
    coco_output = {
        "info": {
            "description": "BUSI Dataset Converted to COCO Format",
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
    
    print(f"Processing {split_name} split with {len(data_list)} images...")
    
    for item in data_list:
        # Create unique filename to avoid collisions if any (though BUSI names are unique per folder)
        # But wait, benign (1).png exists. normal (1).png exists.
        # So we MUST prefix the filename with the type or something unique.
        
        new_filename = f"{item['type']}_{item['filename']}"
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
        height, width = img.shape[:2]
        
        image_id = len(coco_output["images"]) + 1
        
        image_info = {
            "id": image_id,
            "file_name": new_filename,
            "width": width,
            "height": height,
            "date_captured": datetime.now().isoformat(),
            "classification": item['type']
        }
        coco_output["images"].append(image_info)
        
        # Skip annotations for normal images
        if item['type'] == 'normal':
            continue
            
        # Determine Category ID
        if item['type'] == 'benign':
            category_id = 1
        elif item['type'] == 'malignant':
            category_id = 2
        else:
            continue # Should not happen based on logic
            
        # Process masks
        for mask_path in item['mask_paths']:
            mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
                
            # Threshold
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) < 10:
                    continue
                    
                segmentation = contour.flatten().tolist()
                x, y, w, h = cv2.boundingRect(contour)
                bbox = [x, y, w, h]
                area = cv2.contourArea(contour)
                
                # Specific Annotation
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [segmentation],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                coco_output["annotations"].append(annotation)
                annotation_id += 1
                
                # Generic Annotation
                generic_annotation = annotation.copy()
                generic_annotation["id"] = annotation_id
                generic_annotation["category_id"] = 0
                coco_output["annotations"].append(generic_annotation)
                annotation_id += 1

    # Save JSON
    json_path = os.path.join(split_dir, '_annotations.coco.json')
    print(f"Saving COCO JSON to {json_path}...")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=4)

def main():
    # 1. Gather data
    all_data = get_data_list()
    print(f"Total images found: {len(all_data)}")
    
    if not all_data:
        print("No data found!")
        return

    # 2. Split data
    labels = [item['type'] for item in all_data]
    try:
        train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42, stratify=labels)
    except Exception as e:
        print(f"Split error: {e}. Falling back to random split.")
        train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
        
    # 3. Process
    process_split(train_data, 'train')
    process_split(test_data, 'test')
    
    print("Done!")

if __name__ == "__main__":
    main()
