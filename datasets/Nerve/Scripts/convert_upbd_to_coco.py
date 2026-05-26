
import os
import json
import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from datetime import datetime

# Configuration
# Use \\?\ prefix for long paths on Windows
DATASET_ROOT = r"\\?\C:\Users\zhuqh\Desktop\sam3\datasets\神经超声数据集\42.超声臂丛神经(UPBD)"
JPEG_IMAGES_DIR = os.path.join(DATASET_ROOT, "JPEGImages")
JSON_TRAIN_DIR = os.path.join(DATASET_ROOT, "json_train")
OUTPUT_DIR = r"C:\Users\zhuqh\Desktop\sam3\datasets\神经超声数据集\UPBD_coco"

CATEGORIES = [
    {"id": 1, "name": "nerve"},
    {"id": 2, "name": "muscle"},
    {"id": 3, "name": "vein"},
    {"id": 4, "name": "artery"},
]

# Mapping from dataset labels (pinyin) to COCO category names
LABEL_MAPPING = {
    "shenjing": "nerve",
    "jirouzuzhi": "muscle",
    "jingmai": "vein",
    "dongmai": "artery",
    # Ignore others like 'zhifang', 'jizhu'
}

def create_coco_structure():
    return {
        "info": {
            "description": "UPBD Dataset converted to COCO format",
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

def load_json_safe(json_path):
    """Load JSON trying utf-8 then gbk."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        try:
            with open(json_path, 'r', encoding='gbk') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading JSON {json_path} with gbk: {e}")
            return None
    except Exception as e:
        print(f"Error reading JSON {json_path}: {e}")
        return None

def process_dataset():
    all_data = []
    
    # Get list of JSON files
    json_files = glob.glob(os.path.join(JSON_TRAIN_DIR, "*.json"))
    print(f"Found {len(json_files)} JSON files.")

    for json_path in json_files:
        json_filename = os.path.basename(json_path)
        base_name = os.path.splitext(json_filename)[0]
        
        # Try to find corresponding image
        # Assuming .jpg extension as seen in ls output
        image_filename = base_name + ".jpg"
        image_path = os.path.join(JPEG_IMAGES_DIR, image_filename)
        
        if not os.path.exists(image_path):
            # Try to find with other extensions just in case
            possible_extensions = ['.png', '.jpeg', '.bmp']
            found = False
            for ext in possible_extensions:
                temp_path = os.path.join(JPEG_IMAGES_DIR, base_name + ext)
                if os.path.exists(temp_path):
                    image_path = temp_path
                    found = True
                    break
            if not found:
                print(f"Warning: Image not found for {json_filename}")
                continue

        all_data.append({
            "image_path": image_path,
            "json_path": json_path,
            "file_name": os.path.basename(image_path)
        })

    print(f"Found {len(all_data)} valid image-json pairs.")
    
    # Split dataset 80:20
    train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
    
    splits = [("train", train_data), ("test", test_data)]
    
    image_id_counter = 1
    annotation_id_counter = 1
    
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
            
            # Process annotations
            data = load_json_safe(item["json_path"])
            if data is None:
                continue
                
            shapes = data.get("shapes", [])
            for shape in shapes:
                label = shape.get("label", "")
                category_name = LABEL_MAPPING.get(label)
                
                if not category_name:
                    continue
                    
                category_id = next((cat["id"] for cat in CATEGORIES if cat["name"] == category_name), None)
                if category_id is None:
                    continue
                
                points = shape.get("points", [])
                if len(points) < 3:
                    continue
                
                # Flatten points
                segmentation = [coord for point in points for coord in point]
                
                # Calculate BBox and Area
                np_points = np.array(points, dtype=np.float32)
                x, y, w, h = cv2.boundingRect(np_points)
                area = cv2.contourArea(np_points)
                
                coco_output["annotations"].append({
                    "id": annotation_id_counter,
                    "image_id": image_id_counter,
                    "category_id": category_id,
                    "segmentation": [segmentation],
                    "area": float(area),
                    "bbox": [float(x), float(y), float(w), float(h)],
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
