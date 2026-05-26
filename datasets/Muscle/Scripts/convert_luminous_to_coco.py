import os
import cv2
import numpy as np
import json
import shutil
from datetime import datetime
from sklearn.model_selection import train_test_split
import glob

# Config
DATASET_ROOT = r"C:\Users\zhuqh\Desktop\sam3\datasets\肌肉超声数据集\58.LUMINOUS（√）\LUMINOUS_Database"
IMAGES_DIR = os.path.join(DATASET_ROOT, "B-mode")
MASKS_DIR = os.path.join(DATASET_ROOT, "Masks")
OUTPUT_DIR = os.path.join(DATASET_ROOT, "coco_format")

# Categories
# LUMINOUS segments Lumbar Multifidus (LM) Muscle
CATEGORIES = [
    {"id": 1, "name": "multifidus_muscle"}
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

def process_split(file_list, split_name):
    split_dir = create_coco_structure(split_name)
    print(f"Processing {split_name} split with {len(file_list)} images...")
    
    coco_output = {
        "info": {
            "description": f"LUMINOUS Multifidus Muscle Dataset {split_name} Split",
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
    
    for item in file_list:
        src_img_path = item['image_path']
        mask_paths = item['mask_paths'] # List of mask paths
        
        basename = os.path.basename(src_img_path)
        filename_no_ext = os.path.splitext(basename)[0]
        # Rename to png
        png_filename = f"{filename_no_ext}.png"
        
        dst_img_path = os.path.join(split_dir, png_filename)
        
        # Read Image
        try:
            img_data = np.fromfile(src_img_path, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Error reading image {src_img_path}")
                continue
            height, width = img.shape[:2]
            
            # Save as PNG
            # Use cv2.imencode + tofile for unicode paths
            success, buffer = cv2.imencode(".png", img)
            if success:
                with open(dst_img_path, "wb") as f:
                    f.write(buffer)
            else:
                print(f"Failed to encode image {dst_img_path}")
                continue
            
        except Exception as e:
            print(f"Exception reading image {src_img_path}: {e}")
            continue
            
        image_info = {
            "id": current_image_id,
            "file_name": png_filename,
            "width": int(width),
            "height": int(height)
        }
        coco_output['images'].append(image_info)
        
        # Process Masks
        # Combine all masks? Or treat them as separate instances?
        # Usually separate masks mean separate instances or parts.
        # "manually segmented binary masks, serving as the ground truth."
        # Some images have Mask1, Mask2. 
        # These are likely left and right muscles or multiple muscles visible.
        # So we treat each mask file as a separate annotation of the same category.
        
        for mask_path in mask_paths:
            if mask_path and os.path.exists(mask_path):
                try:
                    mask_data = np.fromfile(mask_path, dtype=np.uint8)
                    mask = cv2.imdecode(mask_data, cv2.IMREAD_UNCHANGED)
                    
                    if mask is None:
                        print(f"Error reading mask {mask_path}")
                        continue

                    # If RGB, convert to Gray
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
                            "category_id": 1, # multifidus_muscle
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
        
    print(f"Saved {split_name} annotations to {json_path}")
    print(f"Images: {len(coco_output['images'])}, Annotations: {len(coco_output['annotations'])}")

def get_file_pairs():
    items = []
    
    if not os.path.exists(IMAGES_DIR):
        print(f"Images dir not found: {IMAGES_DIR}")
        return items
        
    # Get all B-mode images
    image_files = glob.glob(os.path.join(IMAGES_DIR, "*_Bmode.tif"))
    
    for img_path in image_files:
        basename = os.path.basename(img_path)
        # Name format: {ID}_{Session}_Bmode.tif
        # Mask format: {ID}_{Session}_Mask.tif OR {ID}_{Session}_Mask1.tif, {ID}_{Session}_Mask2.tif
        
        name_part = basename.replace("_Bmode.tif", "")
        
        # Look for masks
        mask_candidates = []
        
        # Check for simple Mask.tif
        simple_mask = os.path.join(MASKS_DIR, f"{name_part}_Mask.tif")
        if os.path.exists(simple_mask):
            mask_candidates.append(simple_mask)
        
        # Check for Mask1, Mask2, etc.
        # Just globs
        pattern = os.path.join(MASKS_DIR, f"{name_part}_Mask*.tif")
        found_masks = glob.glob(pattern)
        for m in found_masks:
            if m not in mask_candidates:
                mask_candidates.append(m)
                
        if mask_candidates:
            items.append({
                "image_path": img_path,
                "mask_paths": mask_candidates
            })
        else:
            # print(f"Warning: No mask found for {basename}")
            pass
            
    return items

def main():
    print("Scanning dataset...")
    all_items = get_file_pairs()
    print(f"Found {len(all_items)} image-mask pairs.")
    
    if not all_items:
        print("No items found!")
        return
        
    # Split
    train_items, val_items = train_test_split(all_items, test_size=0.2, random_state=42)
    
    process_split(train_items, "train")
    process_split(val_items, "valid")
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()
