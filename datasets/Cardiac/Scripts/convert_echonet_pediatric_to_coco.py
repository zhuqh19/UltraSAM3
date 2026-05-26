import os
import cv2
import json
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

# Dataset paths
DATASET_ROOT = r"C:\Users\zhuqh\Desktop\sam3\datasets\心脏超声数据集\20.EchoNet-Pediatric\echonetpediatric\pediatric_echo_avi"
OUTPUT_DIR = r"C:\Users\zhuqh\Desktop\sam3\datasets\心脏超声数据集\EchoNet_Pediatric_coco"

# COCO categories
# Only Left Ventricle (LV) is labeled in this dataset
CATEGORIES = [
    {"id": 1, "name": "left_ventricle", "supercategory": "heart"},
]

def create_coco_structure():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)

def process_subset(subset_name, view_folder, file_list, tracings_df):
    output_subdir = os.path.join(OUTPUT_DIR, subset_name)
    os.makedirs(output_subdir, exist_ok=True)
    
    images = []
    annotations = []
    annotation_id = 1
    image_id = 1
    
    video_dir = os.path.join(DATASET_ROOT, view_folder, "Videos")
    
    # Filter tracings for current view files
    # view_files set for faster lookup
    view_files_set = set(file_list['FileName'])
    
    # Process each video in the split
    for _, row in tqdm(file_list.iterrows(), total=len(file_list), desc=f"Processing {subset_name} ({view_folder})"):
        filename = row['FileName']
        video_path = os.path.join(video_dir, filename)
        
        if not os.path.exists(video_path):
            # Try to check if it exists in other view folder just in case?
            # But we are processing by view folder.
            continue
            
        # Get tracings for this video
        video_tracings = tracings_df[tracings_df['FileName'] == filename]
        
        if video_tracings.empty:
            continue
            
        # Group by Frame
        # VolumeTracings.csv has X, Y coordinates for each frame
        # We need to reconstruct polygons
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video {video_path}")
            continue
            
        # Get annotated frames
        annotated_frames = video_tracings['Frame'].unique()
        
        for frame_num in annotated_frames:
            # Check if frame_num is numeric
            try:
                frame_idx = float(frame_num)
            except ValueError:
                # print(f"Skipping non-numeric frame index: {frame_num} in {filename}")
                continue

            # Set video to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Get points for this frame
            frame_points = video_tracings[video_tracings['Frame'] == frame_num]
            
            # Points are listed as X1, Y1, X2, Y2... in rows?
            # From preview: FileName, X, Y, Frame
            # So multiple rows per frame.
            
            # Extract polygon points
            polygon_points = []
            for _, point_row in frame_points.iterrows():
                polygon_points.append([point_row['X'], point_row['Y']])
            
            if len(polygon_points) < 3:
                continue
                
            # Convert to numpy for contour ops
            poly_np = np.array(polygon_points).reshape(-1, 2).astype(np.float32)
            
            # Save image
            # Filename: VideoName_FrameNum.jpg
            # Remove .avi from VideoName
            video_stem = os.path.splitext(filename)[0]
            # Ensure frame_num is integer for cleaner filename
            try:
                frame_int = int(float(frame_num))
                img_filename = f"{video_stem}_frame{frame_int}.jpg"
            except:
                img_filename = f"{video_stem}_frame{frame_num}.jpg"
                
            img_output_path = os.path.join(output_subdir, img_filename)
            
            # Use cv2.imencode for unicode path support
            success, encoded_img = cv2.imencode(".jpg", frame)
            if success:
                with open(img_output_path, "wb") as f:
                    f.write(encoded_img)
            else:
                print(f"Failed to encode image: {img_output_path}")
            
            height, width = frame.shape[:2]
            
            # Add image info
            # Use global unique ID across views/splits?
            # Or just local ID. COCO usually wants unique ID.
            # We will use a counter but need to ensure uniqueness if we merge lists later.
            # Here we are processing one subset at a time.
            # To avoid ID conflicts, we can offset IDs or just keep incrementing global ID if we passed it.
            # But here we invoke process_subset multiple times.
            # Let's handle IDs outside or return lists.
            
            current_image_id = image_id # Local to this function call, need better management
            
            images.append({
                "id": current_image_id, # Placeholder, will be re-indexed later
                "file_name": img_filename,
                "width": width,
                "height": height,
                "video_file": filename,
                "frame_index": int(frame_num)
            })
            
            # Add annotation
            # Area and Bbox
            area = cv2.contourArea(poly_np)
            x, y, w, h = cv2.boundingRect(poly_np)
            
            # COCO polygon format: [x1, y1, x2, y2, ...] flat list
            coco_poly = []
            for p in polygon_points:
                coco_poly.extend([float(p[0]), float(p[1])])
                
            annotations.append({
                "id": annotation_id, # Placeholder
                "image_id": current_image_id,
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
    create_coco_structure()
    
    # Process both views
    views = ["A4C", "PSAX"] # Assuming PSAX folder exists and has similar structure
    # Check folders
    available_views = []
    for view in views:
        if os.path.exists(os.path.join(DATASET_ROOT, view)):
            available_views.append(view)
    
    if not available_views:
        print("No view folders found!")
        return
        
    all_images = {"train": [], "test": []}
    all_annotations = {"train": [], "test": []}
    
    global_image_id = 1
    global_annotation_id = 1
    
    for view in available_views:
        print(f"Processing view: {view}")
        
        # Read FileList.csv and VolumeTracings.csv for this view
        # The prompt mentioned "A4C和PSAX两个文件夹，里面有Videos，还有FileList.csv和VolumeTracings.csv"
        # So each view has its own CSVs.
        
        file_list_path = os.path.join(DATASET_ROOT, view, "FileList.csv")
        tracings_path = os.path.join(DATASET_ROOT, view, "VolumeTracings.csv")
        
        if not os.path.exists(file_list_path) or not os.path.exists(tracings_path):
            print(f"Missing CSVs for {view}, skipping.")
            continue
            
        file_list_df = pd.read_csv(file_list_path)
        tracings_df = pd.read_csv(tracings_path)
        
        # Split data
        # FileList.csv has 'Split' column?
        # From preview: FileName,EF,Sex,Age,Weight,Height,Split
        # Split values seen: 5, 7, 1.
        # Need to map Split codes.
        # Usually: TRAIN, VAL, TEST.
        # In EchoNet-Dynamic: Train=0, Val=1, Test=2? Or random?
        # Let's assume standard split logic or use random if not clear.
        # The values 1, 5, 7 seem like fold numbers (k-fold).
        # Let's split by unique patients or just use these folds.
        # Let's define:
        # Train: ~70%, Val: ~15%, Test: ~15%
        # Or just use the Split column if we know the mapping.
        # Without mapping, let's randomly assign Splits to Train/Val/Test.
        
        unique_splits = file_list_df['Split'].unique()
        print(f"Found splits: {unique_splits}")
        
        # If splits are integers (folds), let's distribute them.
        # E.g. sort and take.
        unique_splits.sort()
        # Random shuffle splits
        # random.shuffle(unique_splits) # Keep deterministic for now
        
        # Simple assignment
        n_splits = len(unique_splits)
        # Merge train and val
        # Train: ~80%, Test: ~20%
        n_splits = len(unique_splits)
        n_train = int(n_splits * 0.8)
        
        # Ensure at least 1 for test
        if n_train == n_splits and n_splits > 1:
            n_train = n_splits - 1
        
        train_splits = unique_splits[:n_train]
        test_splits = unique_splits[n_train:]
        
        print(f"Train splits: {train_splits}")
        print(f"Test splits: {test_splits}")
        
        split_map = {
            "train": train_splits,
            "test": test_splits
        }
        
        for subset in ["train", "test"]:
            current_splits = split_map[subset]
            subset_df = file_list_df[file_list_df['Split'].isin(current_splits)]
            
            if subset_df.empty:
                continue
                
            imgs, anns = process_subset(subset, view, subset_df, tracings_df)
            
            # Re-index to ensure global uniqueness
            for img in imgs:
                old_id = img['id']
                img['id'] = global_image_id
                
                # Update corresponding annotations
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
                "description": f"EchoNet-Pediatric Dataset - {subset}",
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
