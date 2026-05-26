import os
import json
import cv2
import numpy as np
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime

# Paths
DATASET_ROOT = r"C:\Users\zhuqh\Desktop\sam3\datasets\肾脏超声数据集\27.肾超声(kidneyUS,申请成功)\kidneyUS_images_25_june_2025"
CSV_PATH = os.path.join(DATASET_ROOT, "reviewed_labels_1.csv")
OUTPUT_DIR = r"C:\Users\zhuqh\Desktop\sam3\datasets\肾脏超声数据集\KidneyUS_coco"

# Categories
# We only focus on "Kidney" (represented by Capsule)
CATEGORIES = [
    {"id": 1, "name": "kidney"}
]

def create_coco_structure(split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    return split_dir

def parse_region_attributes(attr_str):
    try:
        return json.loads(attr_str)
    except:
        return {}

def parse_shape_attributes(shape_str):
    try:
        return json.loads(shape_str)
    except:
        return {}

def get_data_list():
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {CSV_PATH}")
        return []
        
    df = pd.read_csv(CSV_PATH)
    
    # Group by filename
    grouped = df.groupby('filename')
    
    data_list = []
    
    for filename, group in grouped:
        image_path = os.path.join(DATASET_ROOT, filename)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found {filename}")
            continue
            
        # Collect polygons for "Capsule"
        polygons = []
        for _, row in group.iterrows():
            region_attrs = parse_region_attributes(row['region_attributes'])
            shape_attrs = parse_shape_attributes(row['region_shape_attributes'])
            
            # Check if Anatomy is Capsule
            anatomy = region_attrs.get('Anatomy', '')
            if anatomy == 'Capsule':
                if shape_attrs.get('name') == 'polygon':
                    all_points_x = shape_attrs.get('all_points_x', [])
                    all_points_y = shape_attrs.get('all_points_y', [])
                    
                    if len(all_points_x) > 2 and len(all_points_x) == len(all_points_y):
                        # Convert to [[x,y], [x,y]...]
                        points = []
                        for x, y in zip(all_points_x, all_points_y):
                            points.append([x, y])
                        polygons.append(points)
        
        if polygons:
            data_list.append({
                "image_path": image_path,
                "filename": filename,
                "polygons": polygons
            })
            
    return data_list

def process_split(data_list, split_name):
    split_dir = create_coco_structure(split_name)
    print(f"Output directory for {split_name}: {split_dir}")
    
    coco_output = {
        "info": {
            "description": f"KidneyUS Dataset {split_name} Split",
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
            
            # Process Polygons
            for points in item['polygons']:
                # COCO format: [x1, y1, x2, y2, ...]
                segmentation = [coord for point in points for coord in point]
                
                # BBox and Area
                np_points = np.array(points, dtype=np.int32)
                x, y, w, h = cv2.boundingRect(np_points)
                bbox = [x, y, w, h]
                area = cv2.contourArea(np_points)
                
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id_counter,
                    "category_id": 1, # Kidney
                    "segmentation": [segmentation],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                coco_output["annotations"].append(annotation)
                annotation_id += 1
            
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
    print(f"Found {len(all_data)} valid image-annotation pairs.")
    
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
