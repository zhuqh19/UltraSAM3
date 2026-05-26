import cv2
import numpy as np
import os
import glob

# Paths
BASE_DIR = r"\\?\C:\Users\zhuqh\Desktop\sam3\datasets\颈动脉超声数据集\39.CCA\MI_SegNet_dataset"
TS3_LABEL_DIR = os.path.join(BASE_DIR, "TS3", "label")
TRAIN_LABEL_DIR = os.path.join(BASE_DIR, "Training", "label")
VAL_LABEL_DIR = os.path.join(BASE_DIR, "ValS", "label")

def check_labels(dir_path):
    print(f"Checking directory: {dir_path}")
    if not os.path.exists(dir_path):
        print(f"Directory not found: {dir_path}")
        return
        
    label_files = glob.glob(os.path.join(dir_path, "*.png"))
    if not label_files:
        print("No png files found.")
        return
        
    print(f"Found {len(label_files)} label files.")
    
    # Check first few files
    for i in range(min(5, len(label_files))):
        file_path = label_files[i]
        try:
            # Custom read for unicode paths
            stream = open(file_path, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
            
            unique_values = np.unique(img)
            print(f"File: {os.path.basename(file_path)}, Shape: {img.shape}, Unique values: {unique_values}")
            
            if len(unique_values) == 1 and unique_values[0] == 0:
                print("  -> WARNING: Image is all black (all zeros).")
            else:
                print("  -> Image has content.")
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

print("--- Checking TS3 ---")
check_labels(TS3_LABEL_DIR)
print("\n--- Checking Training ---")
check_labels(TRAIN_LABEL_DIR)
print("\n--- Checking ValS ---")
check_labels(VAL_LABEL_DIR)
