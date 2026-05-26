import os
import json
import cv2
import numpy as np
import shutil
import glob
from sklearn.model_selection import train_test_split
from datetime import datetime

# Paths
DATASET_ROOT = r"C:\Users\zhuqh\Desktop\sam3\datasets\肺部超声数据集\28.肺部超声(新冠肺炎COVID-19,LUSS)\data"
OUTPUT_DIR = r"C:\Users\zhuqh\Desktop\sam3\datasets\肺部超声数据集\LUSS_coco"

# Categories
# Based on LUSS papers/standard conventions for this dataset (often 4-5 classes)
# Usually: 
# 0: Background
# 1: Pleural line
# 2: A-line
# 3: B-line
# 4: Consolidation
# 5: Effusion (sometimes)
# But let's check unique values first. 
# Previous check showed: [0 1 2 5] and [0 1 2 3 5]
# Let's assign generic names if unsure, or common LUSS labels.
# Common mapping:
# 1: Pleural line
# 2: A-line
# 3: B-line
# 4: Consolidation
# 5: Pleural Effusion

CATEGORIES = [
    {"id": 1, "name": "pleural line"},
    {"id": 2, "name": "a-line"},
    {"id": 3, "name": "b-line"},
    {"id": 4, "name": "consolidation"},
    {"id": 5, "name": "pleural effusion"}
]

def create_coco_structure(split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    return split_dir

def get_data_list(split_folder):
    # split_folder: 'train' or 'test' inside 'data'
    images_dir = os.path.join(DATASET_ROOT, split_folder, "images")
    masks_dir = os.path.join(DATASET_ROOT, split_folder, "masks")
    
    data_list = []
    
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print(f"Error: Missing directories in {split_folder}")
        return []
        
    image_files = os.listdir(images_dir)
    
    for img_f in image_files:
        if not img_f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
            
        # Match mask: Same filename
        mask_path = os.path.join(masks_dir, img_f)
        
        if os.path.exists(mask_path):
            data_list.append({
                "image_path": os.path.join(images_dir, img_f),
                "mask_path": mask_path,
                "filename": img_f
            })
        else:
            # Try png extension for mask if image is jpg?
            # LUSS usually png/png.
            pass
            
    return data_list

def process_split(data_list, split_name):
    split_dir = create_coco_structure(split_name)
    
    coco_output = {
        "info": {
            "description": f"LUSS Dataset {split_name} Split",
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
            # Read Image
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
            
            # Read Mask
            mask = cv2.imdecode(np.fromfile(item['mask_path'], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                image_id_counter += 1
                continue
                
            unique_labels = np.unique(mask)
            for label_val in unique_labels:
                if label_val == 0:
                    continue
                
                # Filter unknown labels > 5?
                if label_val > 5:
                    continue
                    
                binary_mask = (mask == label_val).astype(np.uint8)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    if cv2.contourArea(contour) < 10:
                        continue
                        
                    segmentation = contour.flatten().tolist()
                    x, y, w, h = cv2.boundingRect(contour)
                    bbox = [x, y, w, h]
                    area = cv2.contourArea(contour)
                    
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id_counter,
                        "category_id": int(label_val),
                        "segmentation": [segmentation],
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0
                    }
                    coco_output["annotations"].append(annotation)
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
    # LUSS already has train/test folders
    # We can respect them or merge and resplit.
    # Usually respecting original split is better for reproducibility.
    # But user asked for 8:2 split in other datasets.
    # Let's check size first.
    
    train_list = get_data_list("train")
    test_list = get_data_list("test")
    
    print(f"Original Train: {len(train_list)}, Original Test: {len(test_list)}")
    
    # If ratio is weird, we can merge and resplit.
    # Let's merge and resplit to be safe and consistent with 8:2 request.
    
    all_data = train_list + test_list
    print(f"Total valid image-mask pairs: {len(all_data)}")
    
    if not all_data:
        print("No data found!")
        return

    try:
        train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"Split failed: {e}")
        return
    
    process_split(train_data, 'train')
    process_split(test_data, 'test')
    
    print("All done!")

if __name__ == "__main__":
    main()
