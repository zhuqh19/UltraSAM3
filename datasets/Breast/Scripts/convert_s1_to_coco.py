import os
import json
import cv2
import numpy as np
import shutil
from datetime import datetime

# Paths
DATASET_ROOT = r"C:\Users\zhuqh\Desktop\sam3\datasets\乳腺超声数据集\34.乳腺超声(S1)\Data"

TRAIN_IMAGES_DIR = os.path.join(DATASET_ROOT, r"TrainingDataSet\BreastTumourImages")
TRAIN_MASKS_DIR = os.path.join(DATASET_ROOT, r"TrainingDataSet\Expanded-3-channel-Labels")

TEST_IMAGES_DIR = os.path.join(DATASET_ROOT, r"TestingDataSet\Test-Expanded-BreastTumourImages")
TEST_MASKS_DIR = os.path.join(DATASET_ROOT, r"TestingDataSet\Test-Expanded-3-channel-Labels")

OUTPUT_DIR = r"C:\Users\zhuqh\Desktop\sam3\datasets\乳腺超声数据集\S1_coco"

# Categories
CATEGORIES = [
    {"id": 0, "name": "breast lesion"},
    {"id": 1, "name": "benign breast tumor"},
    {"id": 2, "name": "malignant breast tumor"}
]

GENERIC_LESION_ID = 0

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

def process_split(images_dir, masks_dir, split_name):
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print(f"Error: Missing directories for {split_name} split.")
        print(f"Images: {images_dir}")
        print(f"Masks: {masks_dir}")
        return

    split_dir = create_coco_structure(split_name)
    
    coco_output = {
        "info": {
            "description": f"S1 Dataset {split_name} Split",
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
    
    image_files = os.listdir(images_dir)
    image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    
    print(f"Processing {split_name} split with {len(image_files)} images...")
    
    annotation_id = 1
    
    for filename in image_files:
        # Construct paths
        image_path = os.path.join(images_dir, filename)
        
        # Mask filename usually matches image filename but extension might differ
        # User said masks are in 'Expanded-3-channel-Labels'
        # Let's check if mask has same basename
        basename = os.path.splitext(filename)[0]
        
        # Try finding mask with common extensions
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            potential_path = os.path.join(masks_dir, basename + ext)
            if os.path.exists(potential_path):
                mask_path = potential_path
                break
        
        if mask_path is None:
            print(f"Warning: Mask not found for {filename}")
            continue
            
        # Copy image
        dst_image_path = os.path.join(split_dir, filename)
        try:
            shutil.copy2(image_path, dst_image_path)
        except Exception as e:
            print(f"Failed to copy {image_path}: {e}")
            continue
            
        # Read image info
        img = read_image(image_path)
        if img is None:
            continue
        height, width = img.shape[:2]
        
        image_id = len(coco_output["images"]) + 1
        
        image_info = {
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
            "date_captured": datetime.now().isoformat()
        }
        coco_output["images"].append(image_info)
        
        # Process Mask
        mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        # Analyze mask color to determine category
        # User said: Dark Gray (Benign), Pure White (Malignant)
        # Verified values: 127 (Benign), 255 (Malignant) based on sampling
        
        unique_values = np.unique(mask)
        
        # Iterate over unique values to find lesion regions
        for val in unique_values:
            if val == 0: # Background
                continue
            
            # Determine Category
            category_id = None
            if 100 <= val <= 150: # Dark Gray ~ 127
                category_id = 1 # Benign
            elif val > 200: # White ~ 255
                category_id = 2 # Malignant
                
            if category_id is None:
                continue
                
            # Create binary mask for this value
            binary_mask = (mask == val).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) < 10:
                    continue
                    
                segmentation = contour.flatten().tolist()
                x, y, w, h = cv2.boundingRect(contour)
                bbox = [x, y, w, h]
                area = cv2.contourArea(contour)
                
                # 1. Specific Annotation
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
                
                # 2. Generic Annotation
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
        
    print(f"Split {split_name} done. Images: {len(coco_output['images'])}, Annotations: {len(coco_output['annotations'])}")

def main():
    # Process Train
    process_split(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, 'train')
    
    # Process Test
    process_split(TEST_IMAGES_DIR, TEST_MASKS_DIR, 'test')
    
    print("All done!")

if __name__ == "__main__":
    main()
