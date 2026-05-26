import os
import cv2
import numpy as np
import json
import shutil
from datetime import datetime
from sklearn.model_selection import train_test_split
import glob

# Config
DATASET_ROOT = r"C:\Users\zhuqh\Desktop\sam3\datasets\肌肉超声数据集\21.小腿肌束超声(FALLMUD)\FALLMUD"
OUTPUT_DIR = os.path.join(DATASET_ROOT, "coco_format")

# Categories
CATEGORIES = [
    {"id": 1, "name": "fascicle"},
    {"id": 2, "name": "aponeurosis"}
]

def create_coco_structure(split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    return split_dir

def binary_mask_to_polygon(binary_mask):
    # Ensure binary 0-1
    binary_mask = (binary_mask > 127).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if contour.size >= 6:
            polygon = contour.flatten().tolist()
            polygons.append(polygon)
    return polygons

def get_bbox(binary_mask):
    binary_mask = (binary_mask > 127).astype(np.uint8)
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return [float(xmin), float(ymin), float(xmax - xmin + 1), float(ymax - ymin + 1)]

def process_split(file_list, split_name):
    split_dir = create_coco_structure(split_name)
    print(f"Processing {split_name} split with {len(file_list)} images...")
    
    coco_output = {
        "info": {
            "description": f"FALLMUD Muscle Ultrasound Dataset {split_name} Split",
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
    
    for item in file_list:
        src_img_path = item['image_path']
        fascicle_mask_path = item['fascicle_mask_path']
        aponeurosis_mask_path = item['aponeurosis_mask_path']
        source_name = item['source'] # RyanCunningham or NeilCronin
        
        # Unique filename
        basename = os.path.basename(src_img_path)
        new_filename = f"{source_name}_{basename}"
        # Ensure extension is consistent or keep original?
        # NeilCronin images are .tif, Ryan are .jpg.
        # Browser/COCO viewers might prefer .jpg or .png.
        # Let's convert everything to .png for consistency and compatibility.
        # Or just keep original.
        # If I convert to png, I need to change extension.
        filename_no_ext = os.path.splitext(basename)[0]
        png_filename = f"{source_name}_{filename_no_ext}.png"
        
        dst_img_path = os.path.join(split_dir, png_filename)
        
        # Read Image
        try:
            img_data = np.fromfile(src_img_path, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                # Try reading with tiffile if cv2 fails for tif?
                # cv2 supports tif usually.
                # Let's try simple imdecode first.
                print(f"Error reading image {src_img_path}")
                continue
            height, width = img.shape[:2]
            
            # Save as PNG
            # Use cv2.imencode + tofile for unicode paths
            success, buffer = cv2.imencode(".png", img)
            if success:
                with open(dst_img_path, "wb") as f:
                    f.write(buffer)
            else:
                print(f"Failed to encode image {dst_img_path}")
                continue
            
        except Exception as e:
            print(f"Exception reading image {src_img_path}: {e}")
            continue
            
        image_info = {
            "id": current_image_id,
            "file_name": png_filename,
            "width": int(width),
            "height": int(height)
        }
        coco_output['images'].append(image_info)
        
        # Process Masks
        masks_to_process = [
            (fascicle_mask_path, 1),      # ID 1: fascicle
            (aponeurosis_mask_path, 2)    # ID 2: aponeurosis
        ]
        
        for mask_path, cat_id in masks_to_process:
            if mask_path and os.path.exists(mask_path):
                try:
                    mask_data = np.fromfile(mask_path, dtype=np.uint8)
                    mask = cv2.imdecode(mask_data, cv2.IMREAD_UNCHANGED)
                    
                    if mask is None:
                        print(f"Error reading mask {mask_path}")
                        continue

                    # If RGB, convert to Gray
                    if len(mask.shape) == 3:
                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    
                    # Resize if needed
                    if mask.shape[:2] != (height, width):
                        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    
                    polygons = binary_mask_to_polygon(mask)
                    bbox = get_bbox(mask)
                    
                    if polygons and bbox:
                        area = float(np.sum(mask > 127))
                        
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
                        
                except Exception as e:
                    print(f"Error processing mask {mask_path}: {e}")
        
        current_image_id += 1
        
    # Save JSON
    json_path = os.path.join(split_dir, '_annotations.coco.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=4)
        
    print(f"Saved {split_name} annotations to {json_path}")
    print(f"Images: {len(coco_output['images'])}, Annotations: {len(coco_output['annotations'])}")

def get_files_from_folder(base_folder, source_name):
    items = []
    images_dir = os.path.join(base_folder, "images")
    fascicle_dir = os.path.join(base_folder, "fascicle_masks")
    aponeurosis_dir = os.path.join(base_folder, "aponeurosis_masks")
    
    if not os.path.exists(images_dir):
        print(f"Images dir not found: {images_dir}")
        return items
        
    # Get all images
    # Need to handle different extensions
    image_files = []
    for ext in ["*.jpg", "*.tif", "*.png", "*.bmp"]:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        
    for img_path in image_files:
        basename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(basename)[0]
        
        # Determine mask paths
        # NeilCronin: images .tif, fascicle .tif, aponeurosis .jpg
        # RyanCunningham: images .jpg, fascicle .jpg, aponeurosis .jpg
        
        fascicle_path = None
        aponeurosis_path = None
        
        if source_name == "NeilCronin":
            # Fascicle is .tif
            f_path = os.path.join(fascicle_dir, name_no_ext + ".tif")
            if os.path.exists(f_path):
                fascicle_path = f_path
            else:
                 # Check jpg just in case
                 f_path_jpg = os.path.join(fascicle_dir, name_no_ext + ".jpg")
                 if os.path.exists(f_path_jpg):
                     fascicle_path = f_path_jpg
            
            # Aponeurosis is .jpg
            a_path = os.path.join(aponeurosis_dir, name_no_ext + ".jpg")
            if os.path.exists(a_path):
                aponeurosis_path = a_path
                
        elif source_name == "RyanCunningham":
            # All .jpg
            f_path = os.path.join(fascicle_dir, name_no_ext + ".jpg")
            if os.path.exists(f_path):
                fascicle_path = f_path
                
            a_path = os.path.join(aponeurosis_dir, name_no_ext + ".jpg")
            if os.path.exists(a_path):
                aponeurosis_path = a_path
                
        items.append({
            "image_path": img_path,
            "fascicle_mask_path": fascicle_path,
            "aponeurosis_mask_path": aponeurosis_path,
            "source": source_name
        })
        
    return items

def main():
    all_items = []
    
    # Process RyanCunningham
    ryan_dir = os.path.join(DATASET_ROOT, "RyanCunningham")
    all_items.extend(get_files_from_folder(ryan_dir, "RyanCunningham"))
    
    # Process NeilCronin
    neil_dir = os.path.join(DATASET_ROOT, "NeilCronin")
    all_items.extend(get_files_from_folder(neil_dir, "NeilCronin"))
    
    print(f"Total items found: {len(all_items)}")
    
    if not all_items:
        print("No items found!")
        return
        
    # Split
    train_items, val_items = train_test_split(all_items, test_size=0.2, random_state=42)
    
    process_split(train_items, "train")
    process_split(val_items, "valid")
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()
