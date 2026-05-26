import os
import json
import cv2
import numpy as np
import shutil
import glob
from pycocotools import mask as maskUtils
from sklearn.model_selection import train_test_split
from datetime import datetime

# Paths
# Use glob to find the directory to handle potential unicode issues or exact naming
DATASET_ROOT_PATTERN = r"C:\Users\zhuqh\Desktop\sam3\datasets\肾脏超声数据集\70*"
matched_dirs = glob.glob(DATASET_ROOT_PATTERN)
if not matched_dirs:
    print("Error: Dataset root not found")
    exit(1)
DATASET_ROOT = matched_dirs[0]
INPUT_DIR = os.path.join(DATASET_ROOT, "Ultrasound Normal KIdney Image.v1i.sam2", "train")
OUTPUT_DIR = r"C:\Users\zhuqh\Desktop\sam3\datasets\肾脏超声数据集\Ultrasound_Normal_Kidney_coco"

# Categories
CATEGORIES = [
    {"id": 1, "name": "kidney"}
]

def create_coco_structure(split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    return split_dir

def get_data_list():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory not found at {INPUT_DIR}")
        return []
        
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    
    data_list = []
    for json_path in json_files:
        # Check corresponding image
        # JSON: name.json -> Image: name.jpg (usually same basename)
        # But here files are like: 10113_jpg.rf.114c... .json and .jpg
        basename = os.path.splitext(os.path.basename(json_path))[0]
        # Try finding image with same basename + .jpg
        image_path = os.path.join(INPUT_DIR, basename + ".jpg")
        
        if not os.path.exists(image_path):
            # Try simply checking if there is a jpg with same name
            # sometimes json is x.jpg.json? No, glob showed x.json
            # Let's check if the json filename *is* the image filename + .json?
            # LS showed: 10113_jpg.rf... .jpg and .json. So just replace extension.
            pass
        
        if os.path.exists(image_path):
            data_list.append({
                "image_path": image_path,
                "json_path": json_path,
                "filename": os.path.basename(image_path)
            })
            
    return data_list

def process_split(data_list, split_name):
    split_dir = create_coco_structure(split_name)
    print(f"Output directory for {split_name}: {split_dir}")
    
    coco_output = {
        "info": {
            "description": f"Ultrasound Normal Kidney Dataset {split_name} Split",
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
            # Read Image for size
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
            
            # Read JSON Annotation
            with open(item['json_path'], 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # JSON format seems to be COCO-like per-image?
            # {"image": {...}, "annotations": [...]}
            # Annotation has "segmentation": {"counts": "...", "size": ...} (RLE)
            
            for ann in data.get('annotations', []):
                # Check segmentation format
                seg = ann.get('segmentation')
                if not seg: continue
                
                # If RLE (dict with counts), decode to mask -> polygons
                if isinstance(seg, dict) and 'counts' in seg:
                    mask = maskUtils.decode(seg)
                    # mask is 0/1 uint8 (Fortran order?)
                    # maskUtils.decode returns (H, W) numpy array, binary
                    
                    # Convert to polygons
                    # Find contours
                    # Need to ensure contiguous array?
                    mask = np.ascontiguousarray(mask)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        if cv2.contourArea(contour) < 10: continue
                        
                        segmentation = contour.flatten().tolist()
                        x, y, w, h = cv2.boundingRect(contour)
                        bbox = [x, y, w, h]
                        area = cv2.contourArea(contour)
                        
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
                        
                elif isinstance(seg, list):
                    # Polygon format
                    # ann['bbox'] might exist
                    bbox = ann.get('bbox', [0,0,0,0])
                    area = ann.get('area', 0)
                    
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id_counter,
                        "category_id": 1,
                        "segmentation": seg,
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
