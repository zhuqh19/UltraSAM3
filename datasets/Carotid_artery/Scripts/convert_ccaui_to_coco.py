import os
import json
import glob
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Configuration ---
# Original dataset path (using long path prefix for Windows)
DATASET_ROOT = r"\\?\C:\Users\zhuqh\Desktop\sam3\datasets\颈动脉超声数据集\16.颈总动脉超声(CCAUI)\Common Carotid Artery Ultrasound Images"
IMAGE_DIR = os.path.join(DATASET_ROOT, "US images")
MASK_DIR = os.path.join(DATASET_ROOT, "Expert mask images")

# Output path
OUTPUT_DIR = r"C:\Users\zhuqh\Desktop\sam3\datasets\颈动脉超声数据集\CCAUI_coco"
# Ensure output path handles long paths if needed
if not OUTPUT_DIR.startswith("\\\\?\\"):
    OUTPUT_DIR = "\\\\?\\" + OUTPUT_DIR

# Categories
CATEGORIES = [
    {"id": 1, "name": "common_carotid_artery"},
]

def cv2_imread(file_path):
    """Read image with unicode path support."""
    try:
        stream = open(file_path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        return cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def cv2_imwrite(file_path, img):
    """Write image with unicode path support."""
    try:
        # Use generated extension based on file_path
        ext = os.path.splitext(file_path)[1]
        result, n = cv2.imencode(ext, img)
        if result:
            with open(file_path, mode='wb') as f:
                n.tofile(f)
            return True
        return False
    except Exception as e:
        print(f"Error writing {file_path}: {e}")
        return False

def create_coco_structure(output_dir):
    """Create COCO directory structure."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for split in ['train', 'test']:
        split_dir = os.path.join(output_dir, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)

def binary_mask_to_polygon(binary_mask):
    """Convert binary mask to COCO polygon format."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentations = []
    
    for contour in contours:
        if contour.size >= 6:  # Need at least 3 points (6 coords)
            contour = contour.flatten().tolist()
            segmentations.append(contour)
            
    return segmentations

def process_dataset():
    print("Scanning dataset...")
    
    # List all image files
    image_files = glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    
    print(f"Found {len(image_files)} images.")
    
    if len(image_files) == 0:
        print("No images found! Check the path.")
        return

    # Split dataset 8:2
    train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)
    
    create_coco_structure(OUTPUT_DIR)
    
    for split, files in [('train', train_files), ('test', test_files)]:
        print(f"Processing {split} split ({len(files)} images)...")
        
        coco_output = {
            "info": {
                "description": "CCAUI Dataset",
                "year": 2024,
                "date_created": "2024-02-24"
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": CATEGORIES
        }
        
        annotation_id = 1
        
        for idx, img_path in enumerate(tqdm(files)):
            # Read image
            img = cv2_imread(img_path)
            if img is None:
                continue
            
            height, width = img.shape[:2]
            file_name = os.path.basename(img_path)
            # COCO usually prefers jpg, but png is fine. Keeping png to avoid compression artifacts on raw US images if desired.
            # However, previous scripts converted to jpg. Let's stick to png as per original dataset or convert?
            # User didn't specify format conversion. The original is PNG. Let's keep PNG or convert to JPG?
            # US images are grayscale/RGB. PNG is lossless. Let's keep PNG for now, but if file size is huge, JPG is better.
            # The previous script converted to JPG. Let's convert to JPG for consistency with other datasets if they were converted.
            # Actually, let's keep original extension or use JPG.
            # Let's use JPG to be safe with typical COCO loaders that expect images.
            file_name_jpg = os.path.splitext(file_name)[0] + ".jpg"
            
            # Copy/Convert image to output directory
            dst_path = os.path.join(OUTPUT_DIR, split, file_name_jpg)
            cv2_imwrite(dst_path, img)
            
            # Add image info
            image_info = {
                "id": idx + 1,
                "file_name": file_name_jpg,
                "width": width,
                "height": height
            }
            coco_output["images"].append(image_info)
            
            # Process mask
            # Mask filename is same as image filename
            mask_path = os.path.join(MASK_DIR, file_name)
            
            if os.path.exists(mask_path):
                mask = cv2_imread(mask_path)
                
                if mask is not None:
                    # Ensure mask is single channel
                    if len(mask.shape) > 2:
                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    
                    # Threshold to binary (0/255)
                    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    
                    # Check if mask has content
                    if cv2.countNonZero(binary_mask) > 0:
                        segmentations = binary_mask_to_polygon(binary_mask)
                        
                        for seg in segmentations:
                            # Calculate bbox and area
                            poly_np = np.array(seg).reshape(-1, 2)
                            x, y, w, h = cv2.boundingRect(poly_np.astype(np.int32))
                            area = cv2.contourArea(poly_np.astype(np.int32))
                            
                            annotation = {
                                "id": annotation_id,
                                "image_id": idx + 1,
                                "category_id": 1,  # common_carotid_artery
                                "segmentation": [seg],
                                "area": area,
                                "bbox": [x, y, w, h],
                                "iscrowd": 0
                            }
                            
                            coco_output["annotations"].append(annotation)
                            annotation_id += 1
            else:
                # print(f"Warning: Mask not found for {file_name}")
                pass
        
        # Save annotations
        json_path = os.path.join(OUTPUT_DIR, split, "_annotations.coco.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_output, f, ensure_ascii=False, indent=4)
            
        print(f"Saved {split} annotations to {json_path}")

if __name__ == "__main__":
    process_dataset()
