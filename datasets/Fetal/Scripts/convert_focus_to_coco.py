import os
import glob
import cv2
import numpy as np
import json
import shutil
from datetime import datetime

# Config
DATASET_ROOT = r"C:\Users\zhuqh\Desktop\sam3\datasets\胎儿超声数据集\71.FOCUS Four-chamber ultrasound image dataset（√）\FOCUS-dataset"
OUTPUT_DIR = os.path.join(DATASET_ROOT, "coco_format")

# Categories
# User request: "segment fetal cardiac"
# Dataset has 'cardiac' and 'thorax' masks.
# Thorax is the chest area, Cardiac is the heart.
# Usually we want the heart.
# Should we include Thorax? 
# "The task is to segment fetal head (FH)-pubic symphysis (PS)" was previous task.
# This task is "segment fetal cardiac".
# I will include both but maybe map them?
# Or just Cardiac?
# Let's include both as separate categories to be safe and comprehensive.
CATEGORIES = [
    {"id": 1, "name": "fetal heart"},
    {"id": 2, "name": "fetal thorax"}
]

def create_coco_structure(split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    return split_dir

def binary_mask_to_polygon(binary_mask):
    # Ensure binary 0-1
    # Mask is 0/255
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

def process_split(split_dirs, output_split_name):
    split_dir = create_coco_structure(output_split_name)
    print(f"Processing {output_split_name} split from {split_dirs}...")
    
    coco_output = {
        "info": {
            "description": f"FOCUS Fetal Cardiac Dataset {output_split_name} Split",
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
    
    # Iterate over source directories (e.g. ['training', 'validation'])
    for source_dir_name in split_dirs:
        source_path = os.path.join(DATASET_ROOT, source_dir_name)
        images_dir = os.path.join(source_path, "images")
        masks_dir = os.path.join(source_path, "annfiles_mask")
        
        if not os.path.exists(images_dir):
            print(f"Images dir not found: {images_dir}")
            continue
            
        # Get images
        image_files = glob.glob(os.path.join(images_dir, "*.png"))
        image_files.sort()
        
        for img_path in image_files:
            basename = os.path.basename(img_path)
            file_id = os.path.splitext(basename)[0] # e.g. "001"
            
            # Copy image
            # To avoid name collision if merging folders (though filenames seem unique per folder? 001.png exists in both?)
            # Yes, training has 001.png, validation likely has 001.png?
            # Let's check. 
            # If filenames are not unique across splits, we should prefix them.
            # But here we are merging training+validation into "train".
            # If training/001.png and validation/001.png exist, they will overwrite!
            # So we MUST prefix with source folder name.
            
            new_filename = f"{source_dir_name}_{basename}"
            dst_img_path = os.path.join(split_dir, new_filename)
            shutil.copy2(img_path, dst_img_path)
            
            # Read Image
            try:
                img_data = np.fromfile(img_path, dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                height, width = img.shape[:2]
            except:
                continue
                
            image_info = {
                "id": current_image_id,
                "file_name": new_filename,
                "width": int(width),
                "height": int(height)
            }
            coco_output['images'].append(image_info)
            
            # Process Masks
            # Pattern: {ID}-cardiac.png and {ID}-thorax.png
            mask_types = [
                ("cardiac", 1), # ID 1
                ("thorax", 2)   # ID 2
            ]
            
            for mask_suffix, cat_id in mask_types:
                mask_filename = f"{file_id}-{mask_suffix}.png"
                mask_path = os.path.join(masks_dir, mask_filename)
                
                if os.path.exists(mask_path):
                    try:
                        mask_data = np.fromfile(mask_path, dtype=np.uint8)
                        mask = cv2.imdecode(mask_data, cv2.IMREAD_UNCHANGED) # Might be RGB (255,255,255)
                        
                        # Convert to single channel if RGB
                        if len(mask.shape) == 3:
                            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                            
                        # Resize if needed
                        if mask.shape[:2] != (height, width):
                            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                            
                        polygons = binary_mask_to_polygon(mask)
                        bbox = get_bbox(mask)
                        
                        if polygons and bbox:
                            area = float(np.sum(mask > 127))
                            
                            ann = {
                                "id": current_annotation_id,
                                "image_id": current_image_id,
                                "category_id": cat_id,
                                "segmentation": polygons,
                                "area": area,
                                "bbox": bbox,
                                "iscrowd": 0
                            }
                            coco_output['annotations'].append(ann)
                            current_annotation_id += 1
                            
                    except Exception as e:
                        print(f"Error processing mask {mask_path}: {e}")
            
            current_image_id += 1

    # Save JSON
    json_path = os.path.join(split_dir, '_annotations.coco.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=4)
        
    print(f"Saved {output_split_name} annotations to {json_path}")
    print(f"Images: {len(coco_output['images'])}, Annotations: {len(coco_output['annotations'])}")

def main():
    # Train = training + validation
    process_split(["training", "validation"], "train")
    
    # Test = testing
    process_split(["testing"], "valid") # Map testing -> valid for COCO standard
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()
