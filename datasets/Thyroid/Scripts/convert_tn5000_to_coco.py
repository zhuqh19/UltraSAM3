import os
import json
import cv2
import numpy as np
import shutil
import glob
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from datetime import datetime

# Paths
DATASET_ROOT = r"C:\Users\zhuqh\Desktop\sam3\datasets\甲状腺超声数据集\64.TN5000（√）\Main data"
IMAGES_DIR = os.path.join(DATASET_ROOT, "JPEGImages")
ANNOTATIONS_DIR = os.path.join(DATASET_ROOT, "Annotations")
OUTPUT_DIR = r"C:\Users\zhuqh\Desktop\sam3\datasets\甲状腺超声数据集\TN5000_coco"

# Categories
# Based on checking XML files, the object name is often "0" or similar generic ID.
# No benign/malignant info found in XML.
CATEGORIES = [
    {"id": 0, "name": "thyroid nodule"},
    {"id": 3, "name": "thyroid tumor"} # Generic specific label
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

def get_data_list():
    data_list = []
    
    if not os.path.exists(IMAGES_DIR) or not os.path.exists(ANNOTATIONS_DIR):
        print(f"Error: Missing directories in {DATASET_ROOT}")
        return []
        
    xml_files = glob.glob(os.path.join(ANNOTATIONS_DIR, "*.xml"))
    
    for xml_path in xml_files:
        # 000001.xml -> 000001.jpg
        basename = os.path.splitext(os.path.basename(xml_path))[0]
        image_filename = basename + ".jpg"
        image_path = os.path.join(IMAGES_DIR, image_filename)
        
        if os.path.exists(image_path):
            data_list.append({
                "image_path": image_path,
                "xml_path": xml_path,
                "filename": image_filename
            })
        else:
            # Try searching for png/jpeg?
            # XML says <filename>000002.jpg</filename>, so likely jpg
            pass
            
    return data_list

def bbox_to_polygon(bbox):
    # bbox: [x, y, w, h]
    x, y, w, h = bbox
    # Polygon: [x, y, x+w, y, x+w, y+h, x, y+h]
    return [x, y, x + w, y, x + w, y + h, x, y + h]

def process_split(data_list, split_name):
    split_dir = create_coco_structure(split_name)
    
    coco_output = {
        "info": {
            "description": f"TN5000 Thyroid Dataset {split_name} Split",
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
        
        # Process XML Annotation
        try:
            tree = ET.parse(item['xml_path'])
            root = tree.getroot()
            
            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    continue
                    
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                w_box = xmax - xmin
                h_box = ymax - ymin
                bbox = [xmin, ymin, w_box, h_box]
                area = w_box * h_box
                
                # Convert BBox to Box-Polygon
                segmentation = [bbox_to_polygon(bbox)]
                
                # 1. Specific Annotation (Thyroid Tumor - ID 3)
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id_counter,
                    "category_id": 3,
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                coco_output["annotations"].append(annotation)
                annotation_id += 1
                
                # 2. Generic Annotation (Thyroid Nodule - ID 0)
                generic_annotation = annotation.copy()
                generic_annotation["id"] = annotation_id
                generic_annotation["category_id"] = 0
                coco_output["annotations"].append(generic_annotation)
                annotation_id += 1
                
        except Exception as e:
            print(f"Error parsing XML {item['xml_path']}: {e}")
        
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
    try:
        train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"Split failed: {e}")
        return
    
    # 3. Process
    process_split(train_data, 'train')
    process_split(test_data, 'test')
    
    print("All done!")

if __name__ == "__main__":
    main()
