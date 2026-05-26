import os
import cv2
import json
import shutil
import numpy as np
from glob import glob
from tqdm import tqdm

# Dataset paths
DATASET_ROOT = r"\\?\C:\Users\zhuqh\Desktop\sam3\datasets\颈动脉超声数据集\39.CCA\MI_SegNet_dataset"
OUTPUT_DIR = r"\\?\C:\Users\zhuqh\Desktop\sam3\datasets\颈动脉超声数据集\CCA_coco"

# COCO categories
CATEGORIES = [
    {"id": 1, "name": "common_carotid_artery", "supercategory": "vessel"},
]

def cv2_imread(file_path):
    """Custom imread to handle Windows paths with unicode."""
    try:
        stream = open(file_path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        return cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def cv2_imwrite(file_path, img):
    """Custom imwrite to handle Windows paths with unicode."""
    try:
        is_success, im_buf_arr = cv2.imencode(".jpg", img)
        im_buf_arr.tofile(file_path)
        return is_success
    except Exception as e:
        print(f"Error writing {file_path}: {e}")
        return False

def create_coco_structure():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)

def binary_mask_to_polygon(mask):
    """Convert binary mask to COCO polygon format."""
    # Ensure mask is binary
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Threshold to ensure binary
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) > 10:  # Filter small noise
            contour = contour.flatten().tolist()
            if len(contour) >= 6:  # Need at least 3 points
                polygons.append(contour)
    
    return polygons

def process_subset(image_list, subset_name):
    print(f"Processing {subset_name} ({len(image_list)} images)...")
    
    images = []
    annotations = []
    annotation_id = 1
    
    output_subdir = os.path.join(OUTPUT_DIR, subset_name)
    
    for img_id, (img_path, mask_path, source_prefix) in enumerate(tqdm(image_list)):
        # Read image
        img = cv2_imread(img_path)
        if img is None:
            continue
            
        height, width = img.shape[:2]
        
        # Read mask
        mask = cv2_imread(mask_path)
        if mask is None:
            print(f"Warning: Mask not found or unreadable: {mask_path}")
            continue
            
        # If mask is 0/1, scale to 0/255 for processing
        if np.max(mask) <= 1:
            mask = mask * 255
            
        # Copy image to destination
        file_name = f"{source_prefix}_{os.path.basename(img_path)}"
        file_name = os.path.splitext(file_name)[0] + ".jpg"
        dest_path = os.path.join(output_subdir, file_name)
        
        # Save as JPG
        cv2_imwrite(dest_path, img)
        
        # Add image info
        image_info = {
            "id": img_id + 1,
            "file_name": file_name,
            "height": height,
            "width": width
        }
        images.append(image_info)
        
        # Process mask
        polygons = binary_mask_to_polygon(mask)
        
        for poly in polygons:
            annotation = {
                "id": annotation_id,
                "image_id": img_id + 1,
                "category_id": 1,
                "segmentation": [poly],
                "area": 0, # Calculate if needed, but often optional for basic training
                "bbox": [], # Calculate if needed
                "iscrowd": 0
            }
            
            # Calculate bbox and area
            poly_np = np.array(poly).reshape((-1, 2))
            x, y, w, h = cv2.boundingRect(poly_np.astype(np.int32))
            annotation["bbox"] = [float(x), float(y), float(w), float(h)]
            annotation["area"] = float(cv2.contourArea(poly_np.astype(np.int32)))
            
            annotations.append(annotation)
            annotation_id += 1
            
    # Save COCO JSON
    coco_output = {
        "info": {
            "description": f"CCA Dataset - {subset_name}",
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

def get_pairs(base_dir, img_folder_name, label_folder_name, source_prefix):
    img_dir = os.path.join(base_dir, img_folder_name)
    label_dir = os.path.join(base_dir, label_folder_name)
    
    pairs = []
    
    if not os.path.exists(img_dir):
        print(f"Warning: {img_dir} does not exist")
        return pairs

    # Check for both png and jpg, though dataset seems to be png
    img_files = glob(os.path.join(img_dir, "*.png"))
    
    for img_path in img_files:
        basename = os.path.basename(img_path)
        
        # Determine label filename based on source
        if "TS3" in source_prefix:
            # Image_X.png -> Label_X.png
            label_basename = basename.replace("Image_", "Label_")
        else:
            # imgXXXX.png -> labelXXXX.png
            label_basename = basename.replace("img", "label")
            
        label_path = os.path.join(label_dir, label_basename)
        
        if os.path.exists(label_path):
            pairs.append((img_path, label_path, source_prefix))
        else:
            print(f"Warning: Label not found for {basename} at {label_path}")
            
    return pairs

def main():
    create_coco_structure()
    
    # Prepare Train sets (Training + ValS)
    train_pairs = []
    
    # 1. Training folder
    train_dir = os.path.join(DATASET_ROOT, "Training")
    train_pairs.extend(get_pairs(train_dir, "img", "label", "Training"))
    
    # 2. ValS folder
    vals_dir = os.path.join(DATASET_ROOT, "ValS")
    train_pairs.extend(get_pairs(vals_dir, "img", "label", "ValS"))
    
    # Prepare Test set (TS3)
    test_pairs = []
    ts3_dir = os.path.join(DATASET_ROOT, "TS3")
    test_pairs.extend(get_pairs(ts3_dir, "img", "label", "TS3"))
    
    print(f"Found {len(train_pairs)} training pairs.")
    print(f"Found {len(test_pairs)} test pairs.")
    
    # Process
    if train_pairs:
        process_subset(train_pairs, "train")
    if test_pairs:
        process_subset(test_pairs, "test")
        
    print("Conversion completed!")

if __name__ == "__main__":
    main()
