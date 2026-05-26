import os
import glob
import cv2
import numpy as np
import json
import shutil
from datetime import datetime
from sklearn.model_selection import train_test_split

# Config
DATASET_ROOT = r"C:\Users\zhuqh\Desktop\sam3\datasets\胎儿超声数据集\23.胎儿脑部超声(Fast-UNet)\Fast-U-Net-main\Dataset"
OUTPUT_DIR = os.path.join(DATASET_ROOT, "coco_format")

# Categories
# Level 1: Head (ID 1), Abdomen (ID 2)
CATEGORIES = [
    {"id": 1, "name": "fetal head"},
    {"id": 2, "name": "fetal abdomen"}
]

HEAD_ID = 1
ABDOMEN_ID = 2

def create_coco_structure(split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    return split_dir

def binary_mask_to_polygon(binary_mask):
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

def process_split(items, split_name):
    split_dir = create_coco_structure(split_name)
    print(f"Processing {split_name} split with {len(items)} images...")
    
    coco_output = {
        "info": {
            "description": f"Fetal Ultrasound Fast-UNet {split_name} Split",
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
    
    for item in items:
        src_img_path = item['image_path']
        src_mask_path = item['mask_path']
        category_id = item['category_id']
        
        # Copy image
        filename = os.path.basename(src_img_path)
        # Handle duplicate filenames across AC/HC?
        # AC has 0001.png, HC has 000_HC.png. They seem distinct.
        # But to be safe, maybe prefix?
        # Actually filenames look distinct enough (HC has _HC).
        
        dst_img_path = os.path.join(split_dir, filename)
        shutil.copy2(src_img_path, dst_img_path)
        
        # Read Image for dims
        # Use numpy fromfile for unicode paths
        try:
            img_data = np.fromfile(src_img_path, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Error reading image {src_img_path}")
                continue
            height, width = img.shape[:2]
        except Exception as e:
            print(f"Exception reading image {src_img_path}: {e}")
            continue
            
        image_info = {
            "id": current_image_id,
            "file_name": filename,
            "width": int(width),
            "height": int(height)
        }
        coco_output['images'].append(image_info)
        
        # Process Mask
        try:
            mask_data = np.fromfile(src_mask_path, dtype=np.uint8)
            mask = cv2.imdecode(mask_data, cv2.IMREAD_UNCHANGED)
            if mask is None:
                print(f"Error reading mask {src_mask_path}")
                continue
                
            # Resize mask if needed? Usually they match.
            if mask.shape[:2] != (height, width):
                print(f"Warning: Mask shape {mask.shape} != Image shape {(height, width)}")
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                
            polygons = binary_mask_to_polygon(mask)
            bbox = get_bbox(mask)
            
            if polygons and bbox:
                area = float(np.sum(mask > 127))
                
                # Annotation
                ann = {
                    "id": current_annotation_id,
                    "image_id": current_image_id,
                    "category_id": category_id,
                    "segmentation": polygons,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                coco_output['annotations'].append(ann)
                current_annotation_id += 1
                
        except Exception as e:
            print(f"Exception processing mask {src_mask_path}: {e}")
            
        current_image_id += 1
        
    # Save JSON
    json_path = os.path.join(split_dir, '_annotations.coco.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=4)
        
    print(f"Saved {split_name} annotations to {json_path}")
    print(f"Images: {len(coco_output['images'])}, Annotations: {len(coco_output['annotations'])}")

def main():
    all_items = []
    
    # 1. Process AC (Abdomen)
    # image1 -> mask
    # image2 -> mask
    ac_dir = os.path.join(DATASET_ROOT, "AC")
    ac_mask_dir = os.path.join(ac_dir, "mask")
    
    for img_folder in ["image1", "image2"]:
        img_dir = os.path.join(ac_dir, img_folder)
        if not os.path.exists(img_dir):
            continue
            
        img_files = glob.glob(os.path.join(img_dir, "*.png"))
        for img_path in img_files:
            basename = os.path.basename(img_path)
            # Mask has same name?
            mask_path = os.path.join(ac_mask_dir, basename)
            if os.path.exists(mask_path):
                all_items.append({
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "category_id": ABDOMEN_ID
                })
            else:
                print(f"Missing mask for AC image: {basename}")

    # 2. Process HC (Head)
    # image1 -> mask (suffix _Annotation)
    hc_dir = os.path.join(DATASET_ROOT, "HC")
    hc_mask_dir = os.path.join(hc_dir, "mask")
    
    # Check image1, image2, image3
    for img_folder in ["image1", "image2", "image3"]:
        img_dir = os.path.join(hc_dir, img_folder)
        if not os.path.exists(img_dir):
            continue
            
        img_files = glob.glob(os.path.join(img_dir, "*.png"))
        for img_path in img_files:
            basename = os.path.basename(img_path)
            name_part = os.path.splitext(basename)[0]
            # Mask name: name_part + "_Annotation.png"
            mask_name = name_part + "_Annotation.png"
            mask_path = os.path.join(hc_mask_dir, mask_name)
            
            if os.path.exists(mask_path):
                all_items.append({
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "category_id": HEAD_ID
                })
            else:
                print(f"Missing mask for HC image: {basename} (Expected: {mask_name})")
                
    print(f"Total items found: {len(all_items)}")
    
    if not all_items:
        return
        
    # Split
    train_items, val_items = train_test_split(all_items, test_size=0.2, random_state=42)
    
    process_split(train_items, "train")
    process_split(val_items, "valid")
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()
