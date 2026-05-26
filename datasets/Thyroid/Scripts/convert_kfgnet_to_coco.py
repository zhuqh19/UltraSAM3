import os
import json
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from datetime import datetime
import glob

# Paths
DATASET_ROOT = r"C:\Users\zhuqh\Desktop\sam3\datasets\甲状腺超声数据集\33.KFGNet（√）\data\images"
OUTPUT_DIR = r"C:\Users\zhuqh\Desktop\sam3\datasets\甲状腺超声数据集\KFGNet_coco"

# Categories
# Filename pattern: videoID_imageID_direction_view_type_frameID.jpg
# type: b (benign), m (malignant)
# Also generic 'thyroid nodule'

CATEGORIES = [
    {"id": 0, "name": "thyroid nodule"},
    {"id": 1, "name": "benign thyroid nodule"},
    {"id": 2, "name": "malignant thyroid nodule"}
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

def parse_filename(filename):
    # Example: 12_0_l_t_b_30.jpg
    # parts: [12, 0, l, t, b, 30]
    # benign/malignant is at index 4 (0-based) or -2 from split by '_' (if extension removed)
    
    basename = os.path.splitext(filename)[0]
    parts = basename.split('_')
    
    if len(parts) >= 5:
        # Check for 'b' or 'm'
        # Usually it's the 5th element (index 4)
        tumor_type = parts[4]
        if tumor_type == 'b':
            return 1 # Benign
        elif tumor_type == 'm':
            return 2 # Malignant
            
    # Fallback or error
    print(f"Warning: Could not determine type from filename {filename}")
    return None

def get_data_list():
    data_list = []
    
    # Get all json files
    json_files = glob.glob(os.path.join(DATASET_ROOT, "*.json"))
    
    for json_path in json_files:
        # Find corresponding image
        # json: name.json -> image: name.jpg
        basename = os.path.splitext(os.path.basename(json_path))[0]
        image_path = os.path.join(DATASET_ROOT, basename + ".jpg")
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found for {json_path}")
            continue
            
        # Parse category
        category_id = parse_filename(os.path.basename(image_path))
        if category_id is None:
            continue
            
        data_list.append({
            "image_path": image_path,
            "json_path": json_path,
            "filename": os.path.basename(image_path),
            "category_id": category_id
        })
        
    return data_list

def process_split(data_list, split_name):
    split_dir = create_coco_structure(split_name)
    
    coco_output = {
        "info": {
            "description": f"KFGNet Thyroid Dataset {split_name} Split",
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
        
        # Process JSON Annotation
        try:
            with open(item['json_path'], 'r') as f:
                data = json.load(f)
                
            for shape in data.get('shapes', []):
                points = shape.get('points', [])
                if not points:
                    continue
                    
                # Points are [[x,y], [x,y]...]
                # COCO segmentation expects [x1, y1, x2, y2, ...] flattened
                segmentation = [coord for point in points for coord in point]
                
                # Calculate BBox and Area
                np_points = np.array(points)
                x_min = np.min(np_points[:, 0])
                y_min = np.min(np_points[:, 1])
                x_max = np.max(np_points[:, 0])
                y_max = np.max(np_points[:, 1])
                
                width_bbox = x_max - x_min
                height_bbox = y_max - y_min
                bbox = [x_min, y_min, width_bbox, height_bbox]
                
                # Area (Shoelace formula via OpenCV)
                # Need integer contour for cv2.contourArea
                contour = np_points.astype(np.float32)
                area = cv2.contourArea(contour)
                
                # 1. Specific Annotation
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id_counter,
                    "category_id": item['category_id'],
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
                
        except Exception as e:
            print(f"Error processing JSON {item['json_path']}: {e}")
        
        image_id_counter += 1

    # Save JSON
    json_path = os.path.join(split_dir, '_annotations.coco.json')
    print(f"Saving COCO JSON to {json_path}...")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=4)
        
    print(f"Split {split_name} done. Images: {len(coco_output['images'])}, Annotations: {len(coco_output['annotations'])}")

def main():
    # 1. Gather data
    all_data = get_data_list()
    print(f"Found {len(all_data)} valid image-annotation pairs.")
    
    if not all_data:
        print("No data found!")
        return

    # 2. Split data (8:2)
    labels = [item['category_id'] for item in all_data]
    try:
        train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42, stratify=labels)
    except Exception as e:
        print(f"Stratified split failed: {e}. Falling back to random split.")
        train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
    
    # 3. Process
    process_split(train_data, 'train')
    process_split(test_data, 'test')
    
    print("All done!")

if __name__ == "__main__":
    main()
