import os
import json
import cv2
import numpy as np
import shutil
import glob
from datetime import datetime

# Paths
# Use glob to find the base directory to handle the special character (√) robustly
BASE_SEARCH = r"C:\Users\zhuqh\Desktop\sam3\datasets\乳腺超声数据集\21.Breast*"
try:
    BASE_DIR = glob.glob(BASE_SEARCH)[0]
    DATASET_ROOT = os.path.join(BASE_DIR, "Miccai 2022 BUV Dataset")
except IndexError:
    print(f"Error: Could not find directory matching {BASE_SEARCH}")
    exit(1)

RAWFRAMES_DIR = os.path.join(DATASET_ROOT, "rawframes")
TRAIN_JSON = os.path.join(DATASET_ROOT, "imagenet_vid_train_15frames.json")
VAL_JSON = os.path.join(DATASET_ROOT, "imagenet_vid_val.json")
OUTPUT_DIR = r"C:\Users\zhuqh\Desktop\sam3\datasets\乳腺超声数据集\BUS_BUV_coco"

# Categories
CATEGORIES = [
    {"id": 0, "name": "breast lesion"},
    {"id": 1, "name": "benign breast tumor"},
    {"id": 2, "name": "malignant breast tumor"}
]

def create_coco_structure(split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    return split_dir

def bbox_to_polygon(bbox):
    # bbox: [x, y, w, h]
    x, y, w, h = bbox
    # Polygon: [x, y, x+w, y, x+w, y+h, x, y+h]
    return [x, y, x + w, y, x + w, y + h, x, y + h]

def process_split(source_json_path, split_name):
    if not os.path.exists(source_json_path):
        print(f"Source JSON not found: {source_json_path}")
        return

    print(f"Processing {split_name} from {source_json_path}...")
    
    with open(source_json_path, 'r') as f:
        source_data = json.load(f)
        
    split_dir = create_coco_structure(split_name)
    
    coco_output = {
        "info": {
            "description": f"BUV Dataset {split_name} Split (Box converted to Mask)",
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
    
    # Map original image_id to annotations
    # source_data['annotations'] is a list of dicts
    annotations_map = {}
    if 'annotations' in source_data:
        for ann in source_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_map:
                annotations_map[img_id] = []
            annotations_map[img_id].append(ann)
            
    # Process images
    # source_data['images'] list of dicts: file_name, id, etc.
    # file_name example: "benign/x28f299ceb056964c/000000.png"
    
    new_image_id = 1
    new_annotation_id = 1
    
    processed_count = 0
    
    for img_info in source_data['images']:
        original_id = img_info['id']
        file_name = img_info['file_name']
        
        # Construct full source path
        # rawframes/benign/...
        source_path = os.path.join(RAWFRAMES_DIR, file_name)
        
        if not os.path.exists(source_path):
            # Try replacing forward slash with backslash just in case
            source_path = os.path.join(RAWFRAMES_DIR, file_name.replace('/', os.sep))
            if not os.path.exists(source_path):
                print(f"Warning: Image not found: {source_path}")
                continue
                
        # Create new filename
        # benign/x28f.../0000.png -> benign_x28f..._0000.png
        safe_name = file_name.replace('/', '_').replace('\\', '_')
        dst_path = os.path.join(split_dir, safe_name)
        
        # Copy image
        try:
            shutil.copy2(source_path, dst_path)
        except Exception as e:
            print(f"Failed to copy {source_path}: {e}")
            continue
            
        # Add to COCO images
        coco_image = {
            "id": new_image_id,
            "file_name": safe_name,
            "width": img_info['width'],
            "height": img_info['height'],
            "date_captured": datetime.now().isoformat(),
            "original_id": original_id
        }
        coco_output["images"].append(coco_image)
        
        # Add annotations
        if original_id in annotations_map:
            for ann in annotations_map[original_id]:
                # Convert bbox to segmentation
                bbox = ann['bbox'] # [x, y, w, h]
                segmentation = [bbox_to_polygon(bbox)]
                area = bbox[2] * bbox[3]
                
                category_id = ann['category_id'] # 1 (benign) or 2 (malignant)
                
                # 1. Specific Annotation
                new_ann = {
                    "id": new_annotation_id,
                    "image_id": new_image_id,
                    "category_id": category_id,
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                coco_output["annotations"].append(new_ann)
                new_annotation_id += 1
                
                # 2. Generic Annotation (Breast Lesion - ID 0)
                generic_ann = new_ann.copy()
                generic_ann["id"] = new_annotation_id
                generic_ann["category_id"] = 0
                coco_output["annotations"].append(generic_ann)
                new_annotation_id += 1
                
        new_image_id += 1
        processed_count += 1
        
        if processed_count % 500 == 0:
            print(f"Processed {processed_count} images...")

    # Save JSON
    json_path = os.path.join(split_dir, '_annotations.coco.json')
    print(f"Saving COCO JSON to {json_path}...")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=4)
        
    print(f"Split {split_name} done. Images: {len(coco_output['images'])}, Annotations: {len(coco_output['annotations'])}")

def main():
    print(f"Dataset Root: {DATASET_ROOT}")
    
    # Process Train
    process_split(TRAIN_JSON, 'train')
    
    # Process Val
    process_split(VAL_JSON, 'test') # User usually maps val to test in this context or keep as val? Let's call it 'test' to match previous structure or 'val'
    # Previous scripts used 'test'. I will use 'test' to match the folder structure I created for others. 
    # Actually, let's create a 'val' folder if it is truly validation, but user asked for train/test usually.
    # The input file is 'imagenet_vid_val.json'. I will map it to 'val' folder to be precise, 
    # BUT consistency with previous 'train/test' structure might be better. 
    # I'll stick to 'train' and 'val' since the input is explicitly named 'val'. 
    # Wait, the user asked for "train and test sets" in previous prompt (convert_bus_dataset_b). 
    # Here they just said "imagenet_vid_train and imagenet_vid_val exist". 
    # I will output to 'train' and 'val' folders.
    
    print("All done!")

if __name__ == "__main__":
    main()
