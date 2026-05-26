import os
import nibabel as nib
import numpy as np
from glob import glob

DATASET_ROOT = r"\\?\C:\Users\zhuqh\Desktop\sam3\datasets\心脏超声数据集\15.心脏超声(CardiacUDC)\archive\cardiacUDC_dataset"

folders = ['label_all_frame', 'Site_G_100', 'Site_G_20', 'Site_G_29', 'Site_R_126', 'Site_R_52', 'Site_R_73']

print("Checking folder counts and label values...")

for folder in folders:
    folder_path = os.path.join(DATASET_ROOT, folder)
    if not os.path.exists(folder_path):
        print(f"Folder {folder} not found.")
        continue
        
    files = glob(os.path.join(folder_path, "*_label.nii.gz"))
    print(f"\nFolder: {folder} - Label files: {len(files)}")
    
    if files:
        # Check first file
        first_file = files[0]
        try:
            nii = nib.load(first_file)
            data = nii.get_fdata()
            unique_values = np.unique(data)
            print(f"  File: {os.path.basename(first_file)}")
            print(f"  Shape: {data.shape}")
            print(f"  Unique values: {unique_values}")
            
            # Check if sparse (for training sets)
            non_zero_frames = []
            if len(data.shape) == 3: # (H, W, T) or similar
                # Assuming last dim is time/frames based on medical imaging conventions usually (H, W, T)
                # But sometimes it's (T, H, W) or (H, W, 1, T)
                # Let's check non-zero slices along last dimension
                for i in range(data.shape[-1]):
                    if np.any(data[..., i] > 0):
                        non_zero_frames.append(i)
                print(f"  Non-zero frames count: {len(non_zero_frames)}")
                if len(non_zero_frames) < 10:
                     print(f"  Non-zero frame indices: {non_zero_frames}")

        except Exception as e:
            print(f"  Error reading {first_file}: {e}")
