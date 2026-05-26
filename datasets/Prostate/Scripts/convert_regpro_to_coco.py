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
DATASET_ROOT = r"C:\Users\zhuqh\Desktop\sam3\datasets\前列腺超声数据集\33.前列腺超声(regPro)\8004388"
OUTPUT_DIR = r"C:\Users\zhuqh\Desktop\sam3\datasets\前列腺超声数据集\RegPro_coco"

# Categories
# Only keep Prostate (Channel 0)
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
    images_dir = os.path.join(DATASET_ROOT, split_folder, "us_images")
    labels_dir = os.path.join(DATASET_ROOT, split_folder, "us_labels")
    
    data_list = []
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Error: Missing directories in {split_folder}")
        return []
        
    nii_files = glob.glob(os.path.join(images_dir, "*.nii.gz"))
    
    for us_path in nii_files:
        basename = os.path.basename(us_path)
        label_path = os.path.join(labels_dir, basename)
        
        if os.path.exists(label_path):
            data_list.append({
                "us_path": us_path,
                "label_path": label_path,
                "basename": basename.replace(".nii.gz", "")
            })
        else:
            print(f"Warning: Label not found for {basename}")
            
    return data_list

def process_volume(item, split_dir, coco_output, image_id_counter, annotation_id_counter):
    try:
        us_nii = nib.load(item['us_path'])
        label_nii = nib.load(item['label_path'])
        
        us_data = us_nii.get_fdata()
        if len(us_data.shape) == 4:
            us_data = np.squeeze(us_data, axis=3)
            
        label_data = label_nii.get_fdata()
        
        depth = us_data.shape[2]
        
        # Check channels
        # If multi-channel, only take channel 0 (Prostate)
        if len(label_data.shape) > 3:
            label_data_prostate = label_data[:, :, :, 0]
        else:
            label_data_prostate = label_data
        
        for z in range(depth):
            us_slice = us_data[:, :, z]
            label_slice = label_data_prostate[:, :, z]
            
            # Skip empty slices
            if np.max(label_slice) == 0:
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
            
            # Process Mask (Channel 0 only)
            # Threshold
            _, binary_mask = cv2.threshold(label_slice.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
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
                    "category_id": 1, # ID 1: Prostate
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
        import traceback
        traceback.print_exc()
        
    return image_id_counter, annotation_id_counter

def process_split(data_list, split_name):
    split_dir = create_coco_structure(split_name)
    print(f"Output directory for {split_name}: {split_dir}")
    
    coco_output = {
        "info": {
            "description": f"RegPro Dataset {split_name} Split",
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
    train_data = get_data_list("train")
    test_data = get_data_list("val")
    
    print(f"Found {len(train_data)} train volumes and {len(test_data)} test volumes.")
    
    if not train_data and not test_data:
        print("No data found!")
        return

    if train_data:
        process_split(train_data, 'train')
    if test_data:
        process_split(test_data, 'test')
    
    print("All done!")

if __name__ == "__main__":
    main()
