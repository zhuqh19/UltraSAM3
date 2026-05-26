import os
import json
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from datetime import datetime
import glob

# Paths
DATASET_ROOT = r"C:\Users\zhuqh\Desktop\sam3\datasets\甲状腺超声数据集\40.甲状腺超声(TN3K)\tn3k"
OUTPUT_DIR = r"C:\Users\zhuqh\Desktop\sam3\datasets\甲状腺超声数据集\TN3K_coco"

# Categories
# No benign/malignant info provided
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

def get_data_list(images_dir, masks_dir):
    data_list = []
    
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print(f"Error: Missing directories: {images_dir} or {masks_dir}")
        return []
        
    image_files = os.listdir(images_dir)
    
    for img_f in image_files:
        if not img_f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
            
        # Match mask
        mask_path = os.path.join(masks_dir, img_f)
        
        if os.path.exists(mask_path):
            data_list.append({
                "image_path": os.path.join(images_dir, img_f),
                "mask_path": mask_path,
                "filename": img_f
            })
            
    return data_list

def process_split(data_list, split_name):
    split_dir = create_coco_structure(split_name)
    
    coco_output = {
        "info": {
            "description": f"TN3K Thyroid Dataset {split_name} Split",
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
        # To avoid name collision if merging later (though not happening here), keep name
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
        # Mask values [0..255]
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
    # Directories
    # TN3K usually has 'trainval-image' and 'test-image'
    # We will merge them and re-split 8:2 as requested
    
    trainval_img = os.path.join(DATASET_ROOT, "trainval-image")
    trainval_mask = os.path.join(DATASET_ROOT, "trainval-mask")
    
    test_img = os.path.join(DATASET_ROOT, "test-image")
    test_mask = os.path.join(DATASET_ROOT, "test-mask")
    
    # 1. Gather all data
    list1 = get_data_list(trainval_img, trainval_mask)
    list2 = get_data_list(test_img, test_mask)
    
    all_data = list1 + list2
    print(f"Found {len(all_data)} valid image-mask pairs (Trainval: {len(list1)}, Test: {len(list2)})")
    
    if not all_data:
        print("No data found!")
        return

    # 2. Split data (8:2)
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
