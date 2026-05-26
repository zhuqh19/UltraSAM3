import os
import json
import cv2
import numpy as np
import shutil
import glob
from sklearn.model_selection import train_test_split
from datetime import datetime

# Paths
DATASET_ROOT_PATTERN = r"C:\Users\zhuqh\Desktop\sam3\datasets\肝脏超声数据集\61*\7272660"
matched_dirs = glob.glob(DATASET_ROOT_PATTERN)
if not matched_dirs:
    print("Error: Dataset root not found")
    exit(1)
DATASET_ROOT = matched_dirs[0]
OUTPUT_DIR = r"C:\Users\zhuqh\Desktop\sam3\datasets\肝脏超声数据集\Annotated_Ultrasound_Liver_coco"

# Categories
# User request: Only segment mass (tumor/lesion).
# Level 1: Liver Lesion (Generic)
# Level 2: Benign, Malignant, Normal (Empty)
# So we will have:
# ID 1: Liver Lesion (Generic class for all masses)
# ID 2: Benign Lesion
# ID 3: Malignant Lesion
# (Normal images will be included but have no annotations)

CATEGORIES = [
    {"id": 1, "name": "liver lesion"},     # Generic
    {"id": 2, "name": "benign lesion"},    # Specific
    {"id": 3, "name": "malignant lesion"}  # Specific
]

def create_coco_structure(split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    return split_dir

def get_data_list():
    data_list = []
    
    classes = ["Benign", "Malignant", "Normal"]
    
    for cls in classes:
        cls_dir = os.path.join(DATASET_ROOT, cls)
        image_dir = os.path.join(cls_dir, "image")
        seg_dir = os.path.join(cls_dir, "segmentation")
        
        if not os.path.exists(image_dir):
            continue
            
        image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
        
        for img_path in image_files:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            
            # We ONLY care about mass segmentation
            mass_json = os.path.join(seg_dir, "mass", f"{basename}.json")
            
            item = {
                "image_path": img_path,
                "filename": f"{cls}_{basename}.jpg",
                "class": cls,
                "mass_json": mass_json if os.path.exists(mass_json) else None
            }
            
            data_list.append(item)
            
    return data_list

def load_polygon_from_json(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            polygon = [coord for point in data for coord in point]
            return polygon, data
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
    return None, None

def process_split(data_list, split_name):
    split_dir = create_coco_structure(split_name)
    print(f"Output directory for {split_name}: {split_dir}")
    
    coco_output = {
        "info": {
            "description": f"Annotated Ultrasound Liver Dataset {split_name} Split",
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
            
            # Process Mass Mask ONLY
            if item['mass_json']:
                poly_flat, poly_points = load_polygon_from_json(item['mass_json'])
                if poly_flat:
                    np_points = np.array(poly_points, dtype=np.int32)
                    x, y, w, h = cv2.boundingRect(np_points)
                    area = cv2.contourArea(np_points)
                    
                    # Determine specific category
                    cat_id_specific = None
                    if item['class'] == 'Benign':
                        cat_id_specific = 2
                    elif item['class'] == 'Malignant':
                        cat_id_specific = 3
                    
                    # 1. Generic Annotation (Liver Lesion - ID 1)
                    annotation_generic = {
                        "id": annotation_id,
                        "image_id": image_id_counter,
                        "category_id": 1,
                        "segmentation": [poly_flat],
                        "area": area,
                        "bbox": [x, y, w, h],
                        "iscrowd": 0
                    }
                    coco_output["annotations"].append(annotation_generic)
                    annotation_id += 1
                    
                    # 2. Specific Annotation (Benign/Malignant - ID 2/3)
                    if cat_id_specific:
                        annotation_specific = annotation_generic.copy()
                        annotation_specific["id"] = annotation_id
                        annotation_specific["category_id"] = cat_id_specific
                        coco_output["annotations"].append(annotation_specific)
                        annotation_id += 1
            
            # Normal images will just have no annotations added
            
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
