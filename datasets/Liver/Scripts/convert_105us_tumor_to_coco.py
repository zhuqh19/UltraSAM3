import os
import json
import cv2
import numpy as np
import shutil
import glob
from sklearn.model_selection import train_test_split
from datetime import datetime

# Paths
DATASET_ROOT = r"C:\Users\zhuqh\Desktop\sam3\datasets\肝脏超声数据集\105US_tumor"
IMAGES_DIR = os.path.join(DATASET_ROOT, "Images")
MASKS_DIR = os.path.join(DATASET_ROOT, "Masks")
OUTPUT_DIR = r"C:\Users\zhuqh\Desktop\sam3\datasets\肝脏超声数据集\105US_tumor_coco"

# Categories
# Level 1: Liver Lesion (Generic)
# Level 2: Tumor (Specific)
# The user asked for:
# 一级标签设置为肝脏病灶 (liver lesion)
# 二级标签设置为肿瘤 (tumor)
# Since we don't have benign/malignant info in filenames (just IDs), we map everything to "tumor".

CATEGORIES = [
    {"id": 1, "name": "liver lesion"},
    {"id": 2, "name": "tumor"}
]

def create_coco_structure(split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    return split_dir

def get_data_list():
    data_list = []
    
    if not os.path.exists(IMAGES_DIR) or not os.path.exists(MASKS_DIR):
        print(f"Error: Missing directories in {DATASET_ROOT}")
        return []
        
    image_files = glob.glob(os.path.join(IMAGES_DIR, "*.png"))
    
    for img_path in image_files:
        basename = os.path.splitext(os.path.basename(img_path))[0] # e.g. "001"
        
        # Mask filename: "001 G man.png"
        mask_filename = f"{basename} G man.png"
        mask_path = os.path.join(MASKS_DIR, mask_filename)
        
        if os.path.exists(mask_path):
            data_list.append({
                "image_path": img_path,
                "mask_path": mask_path,
                "filename": f"{basename}.png"
            })
        else:
            print(f"Warning: Mask not found for {basename}")
            
    return data_list

def process_split(data_list, split_name):
    split_dir = create_coco_structure(split_name)
    print(f"Output directory for {split_name}: {split_dir}")
    
    coco_output = {
        "info": {
            "description": f"105US Tumor Dataset {split_name} Split",
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
            
        try:
            img = cv2.imdecode(np.fromfile(item['image_path'], dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None: continue
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
            # Mask seems to be anti-aliased (many unique values).
            # Histogram shows most pixels are 0, some ~255, and few in between.
            # Threshold at 127 is safe.
            
            mask = cv2.imdecode(np.fromfile(item['mask_path'], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                image_id_counter += 1
                continue
                
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) < 10:
                    continue
                    
                segmentation = contour.flatten().tolist()
                x, y, w, h = cv2.boundingRect(contour)
                bbox = [x, y, w, h]
                area = cv2.contourArea(contour)
                
                # 1. Generic Annotation (Liver Lesion - ID 1)
                annotation_generic = {
                    "id": annotation_id,
                    "image_id": image_id_counter,
                    "category_id": 1,
                    "segmentation": [segmentation],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                coco_output["annotations"].append(annotation_generic)
                annotation_id += 1
                
                # 2. Specific Annotation (Tumor - ID 2)
                annotation_specific = annotation_generic.copy()
                annotation_specific["id"] = annotation_id
                annotation_specific["category_id"] = 2
                coco_output["annotations"].append(annotation_specific)
                annotation_id += 1
            
            image_id_counter += 1
            
        except Exception as e:
            print(f"Error processing {item['filename']}: {e}")

    # Save JSON
    json_path = os.path.join(split_dir, '_annotations.coco.json')
    print(f"Saving COCO JSON to {json_path}...")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=4)
        
    print(f"Split {split_name} done. Images: {len(coco_output['images'])}, Annotations: {len(coco_output['annotations'])}")

def main():
    all_data = get_data_list()
    print(f"Found {len(all_data)} valid images.")
    
    if not all_data:
        print("No data found!")
        return

    # Split 8:2
    try:
        train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"Split failed: {e}")
        return
    
    print(f"Train images: {len(train_data)}, Test images: {len(test_data)}")
    
    process_split(train_data, 'train')
    process_split(test_data, 'test')
    
    print("All done!")

if __name__ == "__main__":
    main()
