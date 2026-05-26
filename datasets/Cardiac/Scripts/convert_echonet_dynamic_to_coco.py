import os
import cv2
import json
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

# Dataset paths
DATASET_ROOT = r"C:\Users\zhuqh\Desktop\sam3\datasets\心脏超声数据集\19.心脏超声(EchoNet-Dynamic)\echonet"
VIDEO_DIR = os.path.join(DATASET_ROOT, "a4c-video-dir", "Videos")
FILE_LIST_PATH = os.path.join(DATASET_ROOT, "a4c-video-dir", "FileList.csv")
TRACINGS_PATH = os.path.join(DATASET_ROOT, "a4c-video-dir", "VolumeTracings.csv")
OUTPUT_DIR = r"C:\Users\zhuqh\Desktop\sam3\datasets\心脏超声数据集\EchoNet_Dynamic_coco"

# COCO categories
CATEGORIES = [
    {"id": 1, "name": "left_ventricle", "supercategory": "heart"},
]

def create_coco_structure():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)

def process_subset(subset_name, file_list, tracings_df):
    output_subdir = os.path.join(OUTPUT_DIR, subset_name)
    os.makedirs(output_subdir, exist_ok=True)
    
    images = []
    annotations = []
    annotation_id = 1
    image_id = 1
    
    # Process each video in the split
    for _, row in tqdm(file_list.iterrows(), total=len(file_list), desc=f"Processing {subset_name}"):
        filename_base = row['FileName']
        filename_avi = f"{filename_base}.avi"
        video_path = os.path.join(VIDEO_DIR, filename_avi)
        
        if not os.path.exists(video_path):
            continue
            
        # Get tracings for this video
        # Tracings have .avi in FileName
        video_tracings = tracings_df[tracings_df['FileName'] == filename_avi]
        
        if video_tracings.empty:
            continue
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video {video_path}")
            continue
            
        # Get annotated frames
        annotated_frames = video_tracings['Frame'].unique()
        
        for frame_num in annotated_frames:
             # Ensure frame_num is integer
            try:
                frame_idx = int(frame_num)
            except ValueError:
                continue

            # Set video to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Get points for this frame
            frame_points = video_tracings[video_tracings['Frame'] == frame_num]
            
            # Sort by index to ensure correct order (assuming CSV order is correct)
            # Typically EchoNet CSVs are ordered
            frame_points = frame_points.sort_index()
            
            if len(frame_points) < 2:
                continue
                
            # Parse rows
            # Row 0: Axis (Apex, BaseCenter) -> We use X1,Y1 as Apex
            # Rows 1..N: Disk Slices (X1,Y1) to (X2,Y2)
            
            rows = [r for _, r in frame_points.iterrows()]
            
            apex = [rows[0]['X1'], rows[0]['Y1']]
            
            left_points = []
            right_points = []
            
            # Skip the first row (Axis) and use the rest for contour
            for r in rows[1:]:
                left_points.append([r['X1'], r['Y1']])
                right_points.append([r['X2'], r['Y2']])
            
            if not left_points:
                continue
                
            # Construct polygon: Apex -> Left -> Right(Reversed) -> Apex
            # Note: Depending on coordinate system, Left/Right might be swapped, but order matters for winding.
            # Usually Apex -> Left down -> Right up -> Apex works.
            
            polygon_points = [apex] + left_points + right_points[::-1]
            
             # Convert to numpy for contour ops
            poly_np = np.array(polygon_points).reshape(-1, 2).astype(np.float32)
            
            # Save image
            img_filename = f"{filename_base}_frame{frame_idx}.jpg"
            img_output_path = os.path.join(output_subdir, img_filename)
            
            success, encoded_img = cv2.imencode(".jpg", frame)
            if success:
                with open(img_output_path, "wb") as f:
                    f.write(encoded_img)
            else:
                continue
            
            height, width = frame.shape[:2]
            
            images.append({
                "id": image_id,
                "file_name": img_filename,
                "width": width,
                "height": height,
                "video_file": filename_avi,
                "frame_index": frame_idx
            })
            
            # Add annotation
            area = cv2.contourArea(poly_np)
            x, y, w, h = cv2.boundingRect(poly_np)
            
            coco_poly = []
            for p in polygon_points:
                coco_poly.extend([float(p[0]), float(p[1])])
                
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [coco_poly],
                "area": float(area),
                "bbox": [float(x), float(y), float(w), float(h)],
                "iscrowd": 0
            })
            
            annotation_id += 1
            image_id += 1
            
        cap.release()
        
    return images, annotations

def main():
    if not os.path.exists(VIDEO_DIR):
        print(f"Error: Video directory not found at {VIDEO_DIR}")
        return

    create_coco_structure()
    
    print("Loading CSVs...")
    file_list_df = pd.read_csv(FILE_LIST_PATH)
    tracings_df = pd.read_csv(TRACINGS_PATH)
    
    # Split mapping
    # EchoNet-Dynamic FileList.csv has Split: TRAIN, VAL, TEST
    # We merge TRAIN and VAL into 'train'
    
    split_map = {
        "train": ["TRAIN", "VAL"],
        "test": ["TEST"]
    }
    
    all_images = {"train": [], "test": []}
    all_annotations = {"train": [], "test": []}
    
    global_image_id = 1
    global_annotation_id = 1
    
    for subset in ["train", "test"]:
        print(f"Processing {subset} set...")
        splits = split_map[subset]
        subset_df = file_list_df[file_list_df['Split'].isin(splits)]
        
        if subset_df.empty:
            print(f"No files found for {subset}")
            continue
            
        imgs, anns = process_subset(subset, subset_df, tracings_df)
        
        # Re-index to ensure global uniqueness (though process_subset starts from 1, we need to accumulate)
        # Actually process_subset uses local IDs starting at 1. We need to offset them.
        
        for img in imgs:
            old_id = img['id']
            img['id'] = global_image_id
            
            for ann in anns:
                if ann['image_id'] == old_id:
                    ann['image_id'] = global_image_id
                    ann['id'] = global_annotation_id
                    global_annotation_id += 1
                    
            global_image_id += 1
            
        all_images[subset].extend(imgs)
        all_annotations[subset].extend(anns)
        
    # Save JSONs
    for subset in ["train", "test"]:
        if not all_images[subset]:
            continue
            
        coco_output = {
            "info": {
                "description": f"EchoNet-Dynamic Dataset - {subset}",
                "year": 2024,
                "date_created": "2024-01-01"
            },
            "images": all_images[subset],
            "annotations": all_annotations[subset],
            "categories": CATEGORIES
        }
        
        json_path = os.path.join(OUTPUT_DIR, subset, "_annotations.coco.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco_output, f, ensure_ascii=False, indent=4)
            
        print(f"Saved {subset} annotations to {json_path}")
        print(f"  Images: {len(all_images[subset])}")
        print(f"  Annotations: {len(all_annotations[subset])}")

    print("Conversion completed!")

if __name__ == "__main__":
    main()
