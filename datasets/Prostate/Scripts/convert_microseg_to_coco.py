import os
import json
import cv2
import numpy as np
import shutil
import glob
import nibabel as nib
from sklearn.model_selection import train_test_split
from datetime import datetime

# Paths
DATASET_ROOT = r"C:\Users\zhuqh\Desktop\sam3\datasets\前列腺超声数据集\29.前列腺超声(MicroSeg，和MUP同一份)\Micro_Ultrasound_Prostate_Segmentation_Dataset"
OUTPUT_DIR = r"C:\Users\zhuqh\Desktop\sam3\datasets\前列腺超声数据集\MicroSeg_coco"

# Categories
# Prostate segmentation usually just has 1 class: Prostate
CATEGORIES = [
    {"id": 1, "name": "prostate"}
]

def create_coco_structure(split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    return split_dir

def save_image(path, img):
    is_success, im_buf = cv2.imencode(".png", img)
    if is_success:
        im_buf.tofile(path)
        return True
    return False

def get_data_list(split_folder):
    # split_folder: 'train' or 'test'
    images_dir = os.path.join(DATASET_ROOT, split_folder, "micro_ultrasound_scans")
    masks_dir = os.path.join(DATASET_ROOT, split_folder, "expert_annotations")
    
    data_list = []
    
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print(f"Error: Missing directories in {split_folder}")
        return []
        
    nii_files = glob.glob(os.path.join(images_dir, "*.nii.gz"))
    
    for us_path in nii_files:
        basename = os.path.basename(us_path)
        # Name: microUS_train_01.nii.gz
        # Mask: expert_annotation_train_01.nii.gz
        
        # Replace prefix 'microUS_' with 'expert_annotation_'
        mask_basename = basename.replace("microUS_", "expert_annotation_")
        mask_path = os.path.join(masks_dir, mask_basename)
        
        if os.path.exists(mask_path):
            data_list.append({
                "us_path": us_path,
                "mask_path": mask_path,
                "basename": basename.replace(".nii.gz", "")
            })
        else:
            print(f"Warning: Mask not found for {basename}")
            
    return data_list

def process_volume(item, split_dir, coco_output, image_id_counter, annotation_id_counter):
    try:
        us_nii = nib.load(item['us_path'])
        mask_nii = nib.load(item['mask_path'])
        
        us_data = us_nii.get_fdata()
        mask_data = mask_nii.get_fdata()
        
        # Shape: (H, W, D)
        depth = us_data.shape[2]
        
        for z in range(depth):
            us_slice = us_data[:, :, z]
            mask_slice = mask_data[:, :, z]
            
            # Skip empty slices
            if np.max(mask_slice) == 0:
                continue
                
            mn, mx = us_slice.min(), us_slice.max()
            if mx > mn:
                us_slice_norm = ((us_slice - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                us_slice_norm = us_slice.astype(np.uint8)
                
            slice_filename = f"{item['basename']}_slice{z:03d}.png"
            slice_path = os.path.join(split_dir, slice_filename)
            
            if not save_image(slice_path, us_slice_norm):
                print(f"Failed to write image: {slice_path}")
                continue
            
            image_info = {
                "id": image_id_counter,
                "file_name": slice_filename,
                "width": us_slice.shape[1],
                "height": us_slice.shape[0],
                "date_captured": datetime.now().isoformat()
            }
            coco_output["images"].append(image_info)
            
            unique_labels = np.unique(mask_slice)
            for label_val in unique_labels:
                if label_val == 0:
                    continue
                    
                binary_mask = (mask_slice == label_val).astype(np.uint8)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    if cv2.contourArea(contour) < 10:
                        continue
                        
                    segmentation = contour.flatten().tolist()
                    x, y, w, h = cv2.boundingRect(contour)
                    bbox = [x, y, w, h]
                    area = cv2.contourArea(contour)
                    
                    annotation = {
                        "id": annotation_id_counter,
                        "image_id": image_id_counter,
                        "category_id": int(label_val),
                        "segmentation": [segmentation],
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0
                    }
                    coco_output["annotations"].append(annotation)
                    annotation_id_counter += 1
            
            image_id_counter += 1
            
    except Exception as e:
        print(f"Error processing {item['basename']}: {e}")
        
    return image_id_counter, annotation_id_counter

def process_split(data_list, split_name):
    split_dir = create_coco_structure(split_name)
    print(f"Output directory for {split_name}: {split_dir}")
    
    coco_output = {
        "info": {
            "description": f"MicroSeg Dataset {split_name} Split",
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
    
    print(f"Processing {split_name} split with {len(data_list)} volumes...")
    
    for item in data_list:
        image_id_counter, annotation_id = process_volume(
            item, split_dir, coco_output, image_id_counter, annotation_id
        )

    json_path = os.path.join(split_dir, '_annotations.coco.json')
    print(f"Saving COCO JSON to {json_path}...")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=4)
        
    print(f"Split {split_name} done. Images: {len(coco_output['images'])}, Annotations: {len(coco_output['annotations'])}")

def main():
    # Gather data from original train/test
    train_list = get_data_list("train")
    test_list = get_data_list("test")
    
    print(f"Original Train: {len(train_list)} volumes")
    print(f"Original Test: {len(test_list)} volumes")
    
    # Merge and resplit 8:2
    all_data = train_list + test_list
    print(f"Total valid volume pairs: {len(all_data)}")
    
    if not all_data:
        print("No data found!")
        return

    try:
        train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"Split failed: {e}")
        return
    
    print(f"Train volumes: {len(train_data)}, Test volumes: {len(test_data)}")
    
    process_split(train_data, 'train')
    process_split(test_data, 'test')
    
    print("All done!")

if __name__ == "__main__":
    main()
