import os
import json
import cv2
import numpy as np
import shutil
import glob
from sklearn.model_selection import train_test_split
from datetime import datetime
from pycocotools import mask as maskUtils  # Optional, but good for RLE if needed. We'll use polygons.

# Configuration
DATASET_ROOT = r"C:\Users\zhuqh\Desktop\sam3\datasets\胎儿超声数据集\22.胎儿腹部结构超声(FASS)\Fetal Abdominal Structures Segmentation Dataset Using Ultrasonic Images"
IMAGES_DIR = os.path.join(DATASET_ROOT, "IMAGES")
MASKS_DIR = os.path.join(DATASET_ROOT, "ARRAY_FORMAT")
OUTPUT_DIR = os.path.join(DATASET_ROOT, "coco_format")

# Categories found in the dataset
# Level 1: Fetal Abdominal Organ (ID 1)
# Level 2: Specific Organs (ID 2-5)
CATEGORIES = [
    {"id": 1, "name": "fetal abdominal organ"},
    {"id": 2, "name": "artery"},
    {"id": 3, "name": "liver"},
    {"id": 4, "name": "stomach"},
    {"id": 5, "name": "vein"}
]

# Map from structure name in .npy to Specific Category ID
CATEGORY_MAP = {
    "artery": 2,
    "liver": 3,
    "stomach": 4,
    "vein": 5
}

GENERIC_CATEGORY_ID = 1

def create_coco_structure(split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    return split_dir

def binary_mask_to_polygon(binary_mask):
    """
    Converts a binary mask to COCO polygon format
    """
    # Ensure binary
    binary_mask = (binary_mask > 0).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if contour.size >= 6:  # Need at least 3 points (6 coords)
            polygon = contour.flatten().tolist()
            polygons.append(polygon)
            
    return polygons

def get_bbox(binary_mask):
    """
    Get bounding box from binary mask [x, y, w, h]
    """
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
            "description": f"Fetal Abdominal Structures Segmentation (FASS) {split_name} Split",
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
    
    current_image_id = 1
    current_annotation_id = 1
    
    for npy_file in file_list:
        basename = os.path.basename(npy_file)
        file_id = os.path.splitext(basename)[0]
        
        # Load data
        try:
            data = np.load(npy_file, allow_pickle=True).item()
        except Exception as e:
            print(f"Error loading {npy_file}: {e}")
            continue
            
        # Get image info
        # Check if corresponding image exists in IMAGES_DIR (prefer png)
        image_filename = file_id + ".png"
        src_image_path = os.path.join(IMAGES_DIR, image_filename)
        
        # If png doesn't exist, check jpg or others, or save from npy
        if not os.path.exists(src_image_path):
            # Try saving from npy
            if 'image' in data:
                img_array = data['image']
                # Determine color space - usually RGB if 3 channels
                # OpenCV uses BGR, so convert if needed. 
                # Assuming data['image'] is RGB (common in loaders), convert to BGR for cv2.imwrite
                # But let's check shape first
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    src_image_path = os.path.join(split_dir, image_filename) # Write directly to dest
                    cv2.imwrite(src_image_path, img_bgr)
                    # We don't need to copy later since we wrote it to dest
                    dst_image_path = src_image_path
                else:
                    print(f"Warning: Image data in {npy_file} has unexpected shape {img_array.shape}")
                    continue
            else:
                print(f"Warning: Image file not found and no image data in npy for {file_id}")
                continue
        else:
            # Copy image to split dir
            dst_image_path = os.path.join(split_dir, image_filename)
            shutil.copy2(src_image_path, dst_image_path)
            
        # Get dimensions
        # If we loaded from npy 'image' key
        if 'image' in data:
            height, width = data['image'].shape[:2]
        else:
            # Read image to get dims
            img = cv2.imread(src_image_path)
            if img is None:
                print(f"Error reading image {src_image_path}")
                continue
            height, width = img.shape[:2]
            
        # Add image info
        image_info = {
            "id": current_image_id,
            "file_name": image_filename,
            "width": int(width),
            "height": int(height)
        }
        coco_output['images'].append(image_info)
        
        # Process annotations
        if 'structures' in data:
            structures = data['structures']
            for struct_name, mask in structures.items():
                if struct_name not in CATEGORY_MAP:
                    continue
                
                category_id = CATEGORY_MAP[struct_name]
                
                # Ensure mask is valid
                if mask is None:
                    continue
                    
                # Get polygons
                polygons = binary_mask_to_polygon(mask)
                
                if not polygons:
                    continue
                
                # For each connected component in the mask, create separate annotations?
                # Or combine all components into one annotation (MultiPolygon)?
                # The user requirement implies hierarchical labeling.
                # Usually, if we have multiple disjoint parts of the SAME organ (e.g. vein branches),
                # they should probably be one annotation if they are the same instance.
                # But here we are doing semantic segmentation -> instance segmentation.
                # Assuming one instance per class per image is a simplification but likely correct for liver/stomach.
                # For veins, it might be multiple. 
                # However, binary_mask_to_polygon returns a LIST of polygons (contours).
                # If we put this list into one "segmentation" field, it is treated as ONE object (iscrowd=0).
                # This matches standard COCO format for single object with multiple parts.
                
                bbox = get_bbox(mask)
                if bbox is None:
                    continue
                    
                area = float(np.sum(mask > 0))
                
                # 1. Specific Annotation (Level 2)
                ann_specific = {
                    "id": current_annotation_id,
                    "image_id": current_image_id,
                    "category_id": category_id,
                    "segmentation": polygons,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                coco_output['annotations'].append(ann_specific)
                current_annotation_id += 1
                
                # 2. Generic Annotation (Level 1 - Fetal Abdominal Organ)
                ann_generic = {
                    "id": current_annotation_id,
                    "image_id": current_image_id,
                    "category_id": GENERIC_CATEGORY_ID,
                    "segmentation": polygons,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                coco_output['annotations'].append(ann_generic)
                current_annotation_id += 1
                
        current_image_id += 1
        
    # Save JSON
    json_path = os.path.join(split_dir, '_annotations.coco.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=4)
        
    print(f"Saved {split_name} annotations to {json_path}")
    print(f"Images: {len(coco_output['images'])}, Annotations: {len(coco_output['annotations'])}")

def main():
    # Find all .npy files
    npy_files = glob.glob(os.path.join(MASKS_DIR, "*.npy"))
    if not npy_files:
        print(f"No .npy files found in {MASKS_DIR}")
        return
        
    print(f"Found {len(npy_files)} files.")
    
    # Split into train/val (80/20)
    train_files, val_files = train_test_split(npy_files, test_size=0.2, random_state=42)
    
    process_split(train_files, "train")
    process_split(val_files, "valid")
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()
