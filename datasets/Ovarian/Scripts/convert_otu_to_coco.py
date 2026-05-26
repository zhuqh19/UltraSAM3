import os
import json
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from datetime import datetime

# Configuration
DATASETS = [
    {
        "name": "OTU_2d",
        "root": r"C:\Users\zhuqh\Desktop\sam3\datasets\卵巢超声数据集\30.卵巢肿瘤超声(MMOTU-2D)\OTU_2d",
        "output_dir": r"C:\Users\zhuqh\Desktop\sam3\datasets\卵巢超声数据集\OTU_2d_coco",
        "mask_suffix": ".PNG",       # Matches 1.JPG -> 1.PNG
        "mask_is_binary_file": False # Use x.PNG
    },
    {
        "name": "OTU_3d",
        "root": r"C:\Users\zhuqh\Desktop\sam3\datasets\卵巢超声数据集\31.卵巢肿瘤超声(MMOTU-3D)\OTU_3d",
        "output_dir": r"C:\Users\zhuqh\Desktop\sam3\datasets\卵巢超声数据集\OTU_3d_coco",
        "mask_suffix": "_binary.PNG", # Matches 1.JPG -> 1_binary.PNG
        "mask_is_binary_file": True   # Use x_binary.PNG
    }
]

# Categories
CATEGORIES = [
    {"id": 0, "name": "ovarian lesion"},
    {"id": 1, "name": "ovarian tumor"}
]

def create_coco_structure(output_dir, split_name):
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    return split_dir

def read_image(path, flags=cv2.IMREAD_COLOR):
    try:
        # Handle Chinese paths or special characters if needed
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    except Exception as e:
        print(f"Error reading image {path}: {e}")
        return None

def process_split(data_list, output_dir, split_name):
    split_dir = create_coco_structure(output_dir, split_name)
    
    coco_output = {
        "info": {
            "description": f"Ovarian Tumor Ultrasound Dataset {split_name} Split",
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
        # Filename: 1.JPG -> 1.JPG
        # To avoid potential conflicts if merged later, we can prefix, but here we keep original names per dataset
        new_filename = item['filename']
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
        mask = read_image(item['mask_path'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            image_id_counter += 1
            continue
            
        # Threshold
        # For 2D: values are ~56. For 3D binary: 0/1.
        # Universal threshold: > 0
        _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 20: # Filter small noise
                continue
                
            segmentation = contour.flatten().tolist()
            x, y, w, h = cv2.boundingRect(contour)
            bbox = [x, y, w, h]
            area = cv2.contourArea(contour)
            
            # 1. Specific Annotation (Ovarian Tumor - ID 1)
            annotation = {
                "id": annotation_id,
                "image_id": image_id_counter,
                "category_id": 1,
                "segmentation": [segmentation],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            }
            coco_output["annotations"].append(annotation)
            annotation_id += 1
            
            # 2. Generic Annotation (Ovarian Lesion - ID 0)
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

def process_dataset(config):
    print(f"\nProcessing Dataset: {config['name']}")
    root_dir = config['root']
    output_dir = config['output_dir']
    mask_suffix = config['mask_suffix']
    
    images_dir = os.path.join(root_dir, "images")
    annotations_dir = os.path.join(root_dir, "annotations")
    
    if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
        print(f"Error: Missing directories in {root_dir}")
        return
        
    # Gather pairs
    data_list = []
    image_files = os.listdir(images_dir)
    
    for img_f in image_files:
        if not img_f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
            
        basename = os.path.splitext(img_f)[0]
        
        # Determine mask filename
        # 2D: 1.JPG -> 1.PNG
        # 3D: 1.JPG -> 1_binary.PNG
        
        mask_filename = basename + mask_suffix
        mask_path = os.path.join(annotations_dir, mask_filename)
        
        if os.path.exists(mask_path):
            data_list.append({
                "image_path": os.path.join(images_dir, img_f),
                "mask_path": mask_path,
                "filename": img_f
            })
        else:
            # Fallback check?
            # User said: 3D has 100_binary.PNG. 2D has 1.PNG.
            # My logic: basename + suffix.
            # 2D: "1" + ".PNG" -> "1.PNG". Correct.
            # 3D: "100" + "_binary.PNG" -> "100_binary.PNG". Correct.
            pass
            
    print(f"Found {len(data_list)} valid image-mask pairs.")
    
    if not data_list:
        print("No data found!")
        return

    # Split 8:2
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    
    # Process
    process_split(train_data, output_dir, 'train')
    process_split(test_data, output_dir, 'test')

def main():
    for config in DATASETS:
        process_dataset(config)
    print("\nAll datasets processed!")

if __name__ == "__main__":
    main()
