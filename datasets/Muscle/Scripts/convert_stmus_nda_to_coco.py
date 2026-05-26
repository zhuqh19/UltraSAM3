
import os
import json
import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from datetime import datetime

# Configuration
# Use \\?\ prefix for long paths on Windows
DATASET_ROOT = r"\\?\C:\Users\zhuqh\Desktop\sam3\datasets\肌肉超声数据集\36.横向肌肉骨骼超声(STMUS_NDA)\DATASET for Deep learning segmentation of transverse musculoskeletal ultrasound images for neuromuscular disease assessment\Polito-Radboud-DeepLearningUS"
OUTPUT_DIR = r"C:\Users\zhuqh\Desktop\sam3\datasets\肌肉超声数据集\STMUS_NDA_coco"

# Define categories based on Muscle Type and Condition
# Structure: Muscle -> Condition
CATEGORIES = [
    {"id": 1, "name": "BB_Healthy", "supercategory": "BB"},
    {"id": 2, "name": "BB_Pathological", "supercategory": "BB"},
    {"id": 3, "name": "GM_Healthy", "supercategory": "GM"},
    {"id": 4, "name": "GM_Pathological", "supercategory": "GM"},
    {"id": 5, "name": "TA_Healthy", "supercategory": "TA"},
    {"id": 6, "name": "TA_Pathological", "supercategory": "TA"},
]

def create_coco_structure():
    return {
        "info": {
            "description": "STMUS_NDA Dataset converted to COCO format",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().strftime("%Y/%m/%d"),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": CATEGORIES,
    }

def cv2_imread(file_path, flags=cv2.IMREAD_COLOR):
    """Read image with unicode path support."""
    try:
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def cv2_imwrite(file_path, img):
    """Write image with unicode path support."""
    try:
        is_success, buffer = cv2.imencode(os.path.splitext(file_path)[1], img)
        if is_success:
            buffer.tofile(file_path)
            return True
        return False
    except Exception as e:
        print(f"Error writing {file_path}: {e}")
        return False

def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) > 4:  # Filter small contours
            contour = contour.flatten().tolist()
            polygons.append(contour)
    return polygons

def process_dataset():
    all_data = []
    image_id_counter = 1
    annotation_id_counter = 1

    muscles = ["BB", "GM", "TA"]
    conditions = ["Healthy", "Pathological"]

    print(f"Scanning dataset at: {DATASET_ROOT}")

    for muscle in muscles:
        for condition in conditions:
            # Determine category ID
            category_name = f"{muscle}_{condition}"
            category_id = next((cat["id"] for cat in CATEGORIES if cat["name"] == category_name), None)
            
            if category_id is None:
                print(f"Warning: No category found for {category_name}")
                continue

            img_dir = os.path.join(DATASET_ROOT, muscle, condition, "Images")
            mask_dir = os.path.join(DATASET_ROOT, muscle, condition, "Masks")

            if not os.path.exists(img_dir):
                print(f"Warning: Directory not found: {img_dir}")
                continue
            
            # List images
            image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in image_files:
                img_path = os.path.join(img_dir, img_file)
                mask_path = os.path.join(mask_dir, img_file) # Assuming same filename

                if not os.path.exists(mask_path):
                    print(f"Warning: Mask not found for {img_file} in {mask_dir}")
                    continue

                all_data.append({
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "category_id": category_id,
                    "file_name": f"{muscle}_{condition}_{img_file}" # Unique filename
                })

    print(f"Found {len(all_data)} valid image-mask pairs.")
    
    # Split dataset
    train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
    
    splits = [("train", train_data), ("test", test_data)]
    
    for split_name, split_data in splits:
        print(f"Processing {split_name} split ({len(split_data)} images)...")
        
        coco_output = create_coco_structure()
        split_dir = os.path.join(OUTPUT_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        for item in split_data:
            # Copy image
            img = cv2_imread(item["image_path"])
            if img is None:
                print(f"Error reading image: {item['image_path']}")
                continue
                
            height, width = img.shape[:2]
            
            dest_img_path = os.path.join(split_dir, item["file_name"])
            if not cv2_imwrite(dest_img_path, img):
                print(f"Error writing image: {dest_img_path}")
                continue
            
            image_info = {
                "id": image_id_counter,
                "file_name": item["file_name"],
                "width": width,
                "height": height
            }
            coco_output["images"].append(image_info)
            
            # Process mask
            mask = cv2_imread(item["mask_path"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Error reading mask: {item['mask_path']}")
                continue
            
            # Threshold mask
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            polygons = mask_to_polygon(binary_mask)
            
            for polygon in polygons:
                coco_output["annotations"].append({
                    "id": annotation_id_counter,
                    "image_id": image_id_counter,
                    "category_id": item["category_id"],
                    "segmentation": [polygon],
                    "area": float(cv2.contourArea(np.array(polygon).reshape(-1, 2))),
                    "bbox": list(cv2.boundingRect(np.array(polygon).reshape(-1, 2))),
                    "iscrowd": 0
                })
                annotation_id_counter += 1
            
            image_id_counter += 1
            
        # Save JSON
        json_path = os.path.join(split_dir, "_annotations.coco.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco_output, f, indent=4)
            
        print(f"Saved {split_name} annotations to {json_path}")

if __name__ == "__main__":
    process_dataset()
