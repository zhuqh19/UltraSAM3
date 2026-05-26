import os
import json
import cv2
import numpy as np
import shutil
from datetime import datetime
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = r"C:\Users\zhuqh\Desktop\sam3\datasets\腹部超声数据集\2.腹部超声(AbdomenUS)\abdominal_US"
OUTPUT_DIR = r"C:\Users\zhuqh\Desktop\sam3\datasets\腹部超声数据集\AbdomenUS_coco"

# Categories
# ID 0: organ (Generic)
# ID 1-8: Specific Organs
CATEGORIES = [
    {"id": 0, "name": "organ"},
    {"id": 1, "name": "liver"},
    {"id": 2, "name": "kidney"},
    {"id": 3, "name": "pancreas"},
    {"id": 4, "name": "vessels"},
    {"id": 5, "name": "adrenals"},
    {"id": 6, "name": "gallbladder"},
    {"id": 7, "name": "bones"},
    {"id": 8, "name": "spleen"}
]

# BGR Colors
COLOR_MAP = {
    "liver":       [100, 0, 100],   # Violet
    "kidney":      [0, 255, 255],   # Yellow
    "pancreas":    [255, 0, 0],     # Blue
    "vessels":     [0, 0, 255],     # Red
    "adrenals":    [255, 255, 0],   # Light Blue (Cyan)
    "gallbladder": [0, 255, 0],     # Green
    "bones":       [255, 255, 255], # White
    "spleen":      [255, 0, 255]    # Pink (Assumed Magenta)
}

# Map category ID to color
ID_TO_COLOR = {
    1: COLOR_MAP["liver"],
    2: COLOR_MAP["kidney"],
    3: COLOR_MAP["pancreas"],
    4: COLOR_MAP["vessels"],
    5: COLOR_MAP["adrenals"],
    6: COLOR_MAP["gallbladder"],
    7: COLOR_MAP["bones"],
    8: COLOR_MAP["spleen"]
}

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

def get_file_pairs(images_dir, annotations_dir):
    pairs = []
    if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
        print(f"Warning: Directory not found: {images_dir} or {annotations_dir}")
        return pairs
        
    # Get all image files
    img_files = os.listdir(images_dir)
    ann_files = os.listdir(annotations_dir)
    
    # Map annotation filenames (without extension) to full path
    ann_map = {os.path.splitext(f)[0]: os.path.join(annotations_dir, f) for f in ann_files}
    
    for img_f in img_files:
        basename = os.path.splitext(img_f)[0]
        if basename in ann_map:
            pairs.append({
                "image_path": os.path.join(images_dir, img_f),
                "mask_path": ann_map[basename],
                "filename": img_f
            })
    return pairs

def color_distance(c1, c2):
    return np.sqrt(np.sum((c1 - c2) ** 2))

def process_split(data_pairs, split_name):
    split_dir = create_coco_structure(split_name)
    
    coco_output = {
        "info": {
            "description": f"AbdomenUS Dataset {split_name} Split",
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
    
    print(f"Processing {split_name} split with {len(data_pairs)} images...")
    
    for item in data_pairs:
        # Determine prefix based on path
        if "AUS" in item['image_path']:
            prefix = "AUS"
        elif "RUS" in item['image_path']:
            prefix = "RUS"
        else:
            prefix = "UNK"
            
        new_filename = f"{prefix}_{item['filename']}"
        dst_image_path = os.path.join(split_dir, new_filename)
        
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
            "file_name": new_filename,
            "width": width,
            "height": height,
            "date_captured": datetime.now().isoformat()
        }
        coco_output["images"].append(image_info)
        
        # Process Mask
        mask = read_image(item['mask_path'], cv2.IMREAD_COLOR)
        if mask is None:
            image_id_counter += 1
            continue
            
        # Analyze unique colors
        unique_colors = np.unique(mask.reshape(-1, 3), axis=0)
        
        for u_color in unique_colors:
            # Skip black (background)
            if np.sum(u_color) < 20: 
                continue
                
            # Find matching category
            matched_cat_id = None
            min_dist = 1000
            
            for cat_id, target_color in ID_TO_COLOR.items():
                dist = color_distance(u_color, np.array(target_color))
                if dist < min_dist:
                    min_dist = dist
                    matched_cat_id = cat_id
            
            # Threshold for color matching
            if min_dist < 40:
                lower = u_color
                upper = u_color
                binary_mask = cv2.inRange(mask, lower, upper)
                
                # Find contours
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    if cv2.contourArea(contour) < 20: # Filter noise
                        continue
                        
                    segmentation = contour.flatten().tolist()
                    x, y, w, h = cv2.boundingRect(contour)
                    bbox = [x, y, w, h]
                    area = cv2.contourArea(contour)
                    
                    # 1. Specific Annotation (e.g., Liver)
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id_counter,
                        "category_id": matched_cat_id,
                        "segmentation": [segmentation],
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0
                    }
                    coco_output["annotations"].append(annotation)
                    annotation_id += 1
                    
                    # 2. Generic Annotation (Organ - ID 0)
                    generic_annotation = annotation.copy()
                    generic_annotation["id"] = annotation_id
                    generic_annotation["category_id"] = 0
                    coco_output["annotations"].append(generic_annotation)
                    annotation_id += 1
        
        image_id_counter += 1

    # Save JSON
    json_path = os.path.join(split_dir, '_annotations.coco.json')
    print(f"Saving COCO JSON to {json_path}...")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=4)
        
    print(f"Split {split_name} done. Images: {len(coco_output['images'])}, Annotations: {len(coco_output['annotations'])}")

def main():
    # Define directories
    aus_root = os.path.join(BASE_DIR, "AUS")
    rus_root = os.path.join(BASE_DIR, "RUS")
    
    # 1. AUS Train
    aus_train_imgs = os.path.join(aus_root, "images", "train")
    aus_train_anns = os.path.join(aus_root, "annotations", "train")
    pairs_aus_train = get_file_pairs(aus_train_imgs, aus_train_anns)
    
    # 2. AUS Test
    aus_test_imgs = os.path.join(aus_root, "images", "test")
    aus_test_anns = os.path.join(aus_root, "annotations", "test")
    pairs_aus_test = get_file_pairs(aus_test_imgs, aus_test_anns)
    
    # 3. RUS Test (to be merged into Train)
    rus_test_imgs = os.path.join(rus_root, "images", "test")
    rus_test_anns = os.path.join(rus_root, "annotations", "test")
    pairs_rus_test = get_file_pairs(rus_test_imgs, rus_test_anns)
    
    # Merge ALL
    all_pairs = pairs_aus_train + pairs_aus_test + pairs_rus_test
    print(f"Total collected images: {len(all_pairs)}")

    if not all_pairs:
        print("No data found!")
        return

    # Random split 8:2
    train_pairs, test_pairs = train_test_split(all_pairs, test_size=0.2, random_state=42)
    
    print(f"Train size: {len(train_pairs)}")
    print(f"Test size: {len(test_pairs)}")

    # Process
    process_split(train_pairs, 'train')
    process_split(test_pairs, 'test')
    
    print("All done!")

if __name__ == "__main__":
    main()
