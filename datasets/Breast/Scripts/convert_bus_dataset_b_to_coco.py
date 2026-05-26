import os
import json
import cv2
import numpy as np
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from datetime import datetime

# Define paths
BASE_DIR = r"C:\Users\zhuqh\Desktop\sam3\datasets\乳腺超声数据集\12.乳腺超声(BUS(DatasetB))\BUS"
IMAGES_DIR = os.path.join(BASE_DIR, "original")
MASKS_DIR = os.path.join(BASE_DIR, "GT")
EXCEL_PATH = os.path.join(BASE_DIR, "DatasetB.xlsx")
OUTPUT_DIR = r"C:\Users\zhuqh\Desktop\sam3\datasets\乳腺超声数据集\BUS_DatasetB_coco"

# Define categories
CATEGORIES = [
    {"id": 0, "name": "breast lesion"},
    {"id": 1, "name": "benign breast tumor"},
    {"id": 2, "name": "malignant breast tumor"}
]

GENERIC_LESION_ID = 0

# Helper function to read image with Chinese path support
def read_image(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)

def create_coco_structure(split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    return split_dir

def process_split(data_list, split_name):
    split_dir = create_coco_structure(split_name)
    
    coco_output = {
        "info": {
            "description": "BUS Dataset B Converted to COCO Format",
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
    
    print(f"Processing {split_name} split with {len(data_list)} images...")
    
    for item in data_list:
        # Copy image
        dst_image_path = os.path.join(split_dir, item['filename'])
        try:
            shutil.copy2(item['image_path'], dst_image_path)
        except Exception as e:
            print(f"Warning: Failed to copy {item['image_path']}: {e}")
            continue
            
        # Read image to get dimensions
        image = read_image(item['image_path'])
        if image is None:
            print(f"Failed to read image: {item['image_path']}, skipping.")
            continue
            
        height, width = image.shape[:2]
        
        # Add image info
        # Use sequential ID for the split
        image_id = len(coco_output["images"]) + 1
        
        image_info = {
            "id": image_id,
            "file_name": item['filename'],
            "width": width,
            "height": height,
            "date_captured": datetime.now().isoformat(),
            "original_id": item['original_id']
        }
        coco_output["images"].append(image_info)
        
        # Process mask
        mask_path = item['mask_path']
        category_id = item['category_id']
        
        if category_id is not None and os.path.exists(mask_path):
            mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Failed to read mask: {mask_path}, skipping annotation.")
                continue
            
            # Threshold mask to binary
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) < 10:  # Filter small noise
                    continue
                    
                # Flatten contour coordinates
                segmentation = contour.flatten().tolist()
                
                # Calculate bounding box
                x, y, w, h = cv2.boundingRect(contour)
                bbox = [x, y, w, h]
                area = cv2.contourArea(contour)
                
                # Add specific annotation (Benign/Malignant)
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [segmentation],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                coco_output["annotations"].append(annotation)
                annotation_id += 1
                
                # Add generic annotation (Breast Lesion - ID 0)
                generic_annotation = annotation.copy()
                generic_annotation["id"] = annotation_id
                generic_annotation["category_id"] = 0
                coco_output["annotations"].append(generic_annotation)
                annotation_id += 1
        
        elif category_id is not None and not os.path.exists(mask_path):
             print(f"Mask not found for image {item['original_id']}: {mask_path}")

    # Save JSON
    json_path = os.path.join(split_dir, '_annotations.coco.json')
    print(f"Saving COCO JSON to {json_path}...")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=4)

def main():
    # Read Excel metadata
    print(f"Reading metadata from {EXCEL_PATH}...")
    try:
        df = pd.read_excel(EXCEL_PATH)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    all_data = []
    
    # Iterate through the dataframe to build data list
    for index, row in df.iterrows():
        try:
            image_id_raw = int(row['Image'])
            tumor_type_str = str(row['Type']).strip()
            
            # Determine category ID
            if tumor_type_str.lower() == 'benign':
                category_id = 1
            elif tumor_type_str.lower() == 'malignant':
                category_id = 2
            else:
                print(f"Unknown type '{tumor_type_str}' for image ID {image_id_raw}, skipping.")
                continue

            # Construct filenames
            filename = f"{image_id_raw:06d}.png"
            image_path = os.path.join(IMAGES_DIR, filename)
            mask_path = os.path.join(MASKS_DIR, filename)
            
            # Check if files exist
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}, skipping.")
                continue
            
            all_data.append({
                "original_id": image_id_raw,
                "filename": filename,
                "image_path": image_path,
                "mask_path": mask_path,
                "category_id": category_id,
                "type": tumor_type_str # for stratification
            })

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue

    print(f"Found {len(all_data)} valid images.")
    
    if not all_data:
        print("No valid data found to process.")
        return

    # Split data
    labels = [item['type'] for item in all_data]
    try:
        train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42, stratify=labels)
    except ValueError as e:
        print(f"Error during splitting (possibly too few samples for stratification): {e}")
        train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)

    # Process splits
    process_split(train_data, 'train')
    process_split(test_data, 'test')
    
    print("Done!")

if __name__ == "__main__":
    main()
