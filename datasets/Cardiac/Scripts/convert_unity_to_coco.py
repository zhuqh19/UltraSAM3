import os
import cv2
import json
import shutil
import numpy as np
from glob import glob
from tqdm import tqdm
import random

# Dataset paths
DATASET_ROOT = r"C:\Users\zhuqh\Desktop\sam3\datasets\心脏超声数据集\52.Unity Imaging Colloborative"
LABELS_FILE = os.path.join(DATASET_ROOT, "u4s-labels", "labels-all.json")
IMG_DIR = os.path.join(DATASET_ROOT, "png-cache")
OUTPUT_DIR = os.path.join(DATASET_ROOT, "Unity_coco")

# COCO categories
# User only wants "Left ventricular endocardial contour"
CATEGORIES = [
    {"id": 1, "name": "left_ventricle", "supercategory": "heart"},
]

def create_coco_structure():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)

def parse_curve_string(x_str, y_str):
    """Parse space-separated x and y coordinate strings into a polygon list."""
    if not x_str or not y_str:
        return []
    
    try:
        xs = [float(x) for x in x_str.strip().split()]
        ys = [float(y) for y in y_str.strip().split()]
        
        if len(xs) != len(ys) or len(xs) < 3:
            return []
        
        # Combine into [x1, y1, x2, y2, ...]
        polygon = []
        for x, y in zip(xs, ys):
            polygon.append(x)
            polygon.append(y)
            
        return [polygon]
    except ValueError:
        return []

def process_dataset(data, img_files, subset_name, image_map):
    output_subdir = os.path.join(OUTPUT_DIR, subset_name)
    os.makedirs(output_subdir, exist_ok=True)
    
    images = []
    annotations = []
    annotation_id = 1
    image_id = 1
    
    # Filter data to only include files in the current split
    split_data = {k: v for k, v in data.items() if k in img_files}
    
    for filename, entry in tqdm(split_data.items(), desc=f"Processing {subset_name}"):
        try:
            img_path = image_map[filename]
            
            # Check if image exists
            if not os.path.exists(img_path):
                # Try finding it recursively if flat structure assumption is wrong
                # But for now assume flat structure in png-cache
                continue
                
            # Read image to get dimensions
            # Use cv2.imdecode to handle unicode paths if needed, though standard open is fine for reading
            # But cv2.imread might fail with unicode on Windows, so utilize numpy
            # img = cv2.imread(img_path)
            # Safe read:
            with open(img_path, "rb") as f:
                bytes_data = bytearray(f.read())
                numpy_array = np.asarray(bytes_data, dtype=np.uint8)
                img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"Warning: Could not read image {filename}")
                continue
                
            height, width = img.shape[:2]
            
            # Save image to output dir
            # We can copy the file directly to avoid re-encoding loss, 
            # but usually COCO datasets have clean filenames.
            # Let's keep original filename for traceability
            shutil.copy2(img_path, os.path.join(output_subdir, filename))
            
            images.append({
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height
            })
            
            # Process labels
            labels = entry.get("labels", {})
            
            # Extract curve-lv-endo
            if "curve-lv-endo" in labels:
                contours = labels["curve-lv-endo"]
                if not isinstance(contours, list):
                    contours = [contours]
                    
                for contour in contours:
                    # Check validity
                    if contour.get("type") == "off":
                        continue
                        
                    x_str = contour.get("x", "")
                    y_str = contour.get("y", "")
                    
                    polygons = parse_curve_string(x_str, y_str)
                    
                    for poly in polygons:
                        # Calculate area and bbox
                        # Poly is [x1, y1, x2, y2, ...]
                        poly_np = np.array(poly).reshape(-1, 2).astype(np.float32)
                        
                        # COCO requires area and bbox
                        area = cv2.contourArea(poly_np)
                        x, y, w, h = cv2.boundingRect(poly_np)
                        
                        annotations.append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 1, # left_ventricle
                            "segmentation": [poly],
                            "area": area,
                            "bbox": [x, y, w, h],
                            "iscrowd": 0
                        })
                        annotation_id += 1
            
            image_id += 1
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    # Save COCO JSON
    coco_output = {
        "info": {
            "description": f"Unity Imaging Dataset - {subset_name}",
            "year": 2024,
            "date_created": "2024-01-01"
        },
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES
    }
    
    json_path = os.path.join(output_subdir, "_annotations.coco.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=4)
        
    print(f"Saved {subset_name} annotations to {json_path}")
    print(f"Processed {len(images)} images and {len(annotations)} annotations.")

def main():
    if not os.path.exists(IMG_DIR):
        print(f"Error: Image directory not found at {IMG_DIR}")
        print("Please ensure 'png-cache' folder exists and contains the images.")
        # Create directory to avoid crash if user wants to put images there later?
        # No, better to fail or just return.
        return

    if not os.path.exists(LABELS_FILE):
        print(f"Error: Labels file not found at {LABELS_FILE}")
        return
        
    create_coco_structure()
    
    print("Loading labels...")
    with open(LABELS_FILE, "r") as f:
        data = json.load(f)
        
    # Get all filenames from JSON that have labels
    # Actually keys are filenames.
    all_filenames = list(data.keys())
    
    # Create a map of filename -> full path
    print("Scanning for images recursively...")
    image_map = {}
    for root, dirs, files in os.walk(IMG_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_map[file] = os.path.join(root, file)
    
    print(f"Found {len(image_map)} images in {IMG_DIR} and subdirectories.")

    # Filter valid images (those that exist)
    valid_filenames = []
    print("Checking for existing images...")
    for fname in tqdm(all_filenames):
        if fname in image_map:
            valid_filenames.append(fname)
            
    print(f"Found {len(valid_filenames)} images out of {len(all_filenames)} labels.")
    
    if not valid_filenames:
        print("No matching images found in png-cache. Exiting.")
        return
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(valid_filenames)
    
    split_idx = int(len(valid_filenames) * 0.8)
    train_files = set(valid_filenames[:split_idx])
    test_files = set(valid_filenames[split_idx:])
    
    print(f"Train files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")
    
    process_dataset(data, train_files, "train", image_map)
    process_dataset(data, test_files, "test", image_map)
    
    print("Conversion completed!")

if __name__ == "__main__":
    main()
