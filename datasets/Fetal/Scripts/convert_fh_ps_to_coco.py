import SimpleITK as sitk
import numpy as np
import os
import shutil
import json
import glob
from sklearn.model_selection import train_test_split
from datetime import datetime
import cv2

# Config
DATASET_ROOT = r"C:\Users\zhuqh\Desktop\sam3\datasets\胎儿超声数据集\24.FH-PS-AOP(申请成功下载完成)\Pubic Symphysis-Fetal Head Segmentation and Angle of Progression"
IMAGE_DIR = os.path.join(DATASET_ROOT, "image_mha")
LABEL_DIR = os.path.join(DATASET_ROOT, "label_mha")
OUTPUT_DIR = os.path.join(DATASET_ROOT, "coco_format")

# Categories
# User request: "fetal head (FH)-pubic symphysis (PS)"
# Label file has values [0, 1, 2].
# Need to determine which is which. 
# Usually in medical segmentation:
# 1 is FH (Fetal Head), 2 is PS (Pubic Symphysis) or vice versa.
# Given FH is usually larger and the main object, let's assume 1=FH, 2=PS?
# Or we can check the dataset description or just assign them.
# The user prompt says "segment fetal head (FH)-pubic symphysis (PS)".
# Let's define categories as:
# 1: fetal head
# 2: pubic symphysis
CATEGORIES = [
    {"id": 1, "name": "fetal head"},
    {"id": 2, "name": "pubic symphysis"}
]

def create_coco_structure(split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    return split_dir

def binary_mask_to_polygon(binary_mask):
    binary_mask = (binary_mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if contour.size >= 6:
            polygon = contour.flatten().tolist()
            polygons.append(polygon)
    return polygons

def get_bbox(binary_mask):
    binary_mask = (binary_mask > 0).astype(np.uint8)
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return [float(xmin), float(ymin), float(xmax - xmin + 1), float(ymax - ymin + 1)]

def read_mha_image(path):
    # Use temp file for unicode safety
    # Add random number to temp path to avoid conflict
    temp_path = f"temp_img_{os.getpid()}.mha"
    try:
        shutil.copy2(path, temp_path)
        img = sitk.ReadImage(temp_path)
        data = sitk.GetArrayFromImage(img) # (Channels, H, W) or (H, W)?
        
        # Check shape from inspect output: (3, 256, 256) for image -> Channels first
        # But this dataset is ultrasound, usually grayscale.
        # If it is (3, 256, 256), it might be treated as RGB.
        # Transpose to (H, W, C) for OpenCV.
        if len(data.shape) == 3:
            if data.shape[0] == 3: # (3, H, W) -> (H, W, 3)
                data = np.transpose(data, (1, 2, 0))
            # If shape[0] is not 3, maybe it is (Slices, H, W)?
            # 256 slices? Unlikely for 2D.
            # If it is (1, H, W), squeeze it.
            elif data.shape[0] == 1:
                 data = np.squeeze(data, axis=0)
        
        return data
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

def read_mha_label(path):
    temp_path = f"temp_lbl_{os.getpid()}.mha"
    try:
        shutil.copy2(path, temp_path)
        img = sitk.ReadImage(temp_path)
        data = sitk.GetArrayFromImage(img)
        # Inspect said Shape (numpy): (256, 256) for label.
        return data
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

def process_split(file_list, split_name):
    split_dir = create_coco_structure(split_name)
    print(f"Processing {split_name} split with {len(file_list)} images...")
    
    # Try to ensure split_dir exists and is writable
    if not os.path.exists(split_dir):
        os.makedirs(split_dir, exist_ok=True)
        
    coco_output = {
        "info": {
            "description": f"FH-PS-AOP Dataset {split_name} Split",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": CATEGORIES
    }
    
    current_image_id = 1
    current_annotation_id = 1
    
    for mha_file in file_list:
        basename = os.path.basename(mha_file)
        file_id = os.path.splitext(basename)[0]
        
        src_img_path = os.path.join(IMAGE_DIR, basename)
        src_lbl_path = os.path.join(LABEL_DIR, basename)
        
        if not os.path.exists(src_lbl_path):
            print(f"Label not found for {basename}")
            continue
            
        # Read Image
        img_data = read_mha_image(src_img_path)
        if img_data is None:
            continue
            
        height, width = img_data.shape[:2]
        
        # Save as PNG
        png_filename = file_id + ".png"
        dst_img_path = os.path.join(split_dir, png_filename)
        
        # Ensure directory exists (just in case)
        if not os.path.exists(os.path.dirname(dst_img_path)):
            os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
            
        # Ensure data is uint8
        if img_data.dtype != np.uint8:
            img_data = (img_data / np.max(img_data) * 255).astype(np.uint8)
            
        # If it's RGB, convert to BGR for OpenCV
        if len(img_data.shape) == 3 and img_data.shape[2] == 3:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            
        # Use imencode and tofile to support unicode paths
        is_success, im_buf_arr = cv2.imencode(".png", img_data)
        if is_success:
            try:
                im_buf_arr.tofile(dst_img_path)
                success = True
            except Exception as e:
                print(f"Write failed: {e}")
                success = False
        else:
            success = False

        if not success:
            print(f"Failed to write image: {dst_img_path}")
            continue
        
        image_info = {
            "id": current_image_id,
            "file_name": png_filename,
            "width": int(width),
            "height": int(height)
        }
        coco_output['images'].append(image_info)
        
        # Process Label
        lbl_data = read_mha_label(src_lbl_path)
        if lbl_data is None:
            continue
            
        # Categories: 1=FH, 2=PS
        for cat in CATEGORIES:
            cat_id = cat['id']
            # Create binary mask for this category
            binary_mask = (lbl_data == cat_id).astype(np.uint8)
            
            polygons = binary_mask_to_polygon(binary_mask)
            bbox = get_bbox(binary_mask)
            
            if polygons and bbox:
                area = float(np.sum(binary_mask))
                
                ann = {
                    "id": current_annotation_id,
                    "image_id": current_image_id,
                    "category_id": cat_id,
                    "segmentation": polygons,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                coco_output['annotations'].append(ann)
                current_annotation_id += 1
                
        current_image_id += 1
        
    # Save JSON
    json_path = os.path.join(split_dir, '_annotations.coco.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=4)
        
    print(f"Saved {split_name} annotations to {json_path}")
    print(f"Images: {len(coco_output['images'])}, Annotations: {len(coco_output['annotations'])}")

def main():
    # Find all mha files
    mha_files = glob.glob(os.path.join(IMAGE_DIR, "*.mha"))
    if not mha_files:
        print("No mha files found")
        return
        
    print(f"Found {len(mha_files)} files.")
    
    # Split
    train_files, val_files = train_test_split(mha_files, test_size=0.2, random_state=42)
    
    process_split(train_files, "train")
    process_split(val_files, "valid")
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()
