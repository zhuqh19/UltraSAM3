import nibabel as nib
import numpy as np
import os
import glob

dataset_path = r"C:\Users\zhuqh\Desktop\sam3\datasets\心脏超声数据集\18.心脏超声(EchoCP)\archive\EchoCP_dataset"
files = glob.glob(os.path.join(dataset_path, "*_label.nii.gz"))[:5]

for f in files:
    try:
        nii = nib.load(f)
        data = nii.get_fdata()
        unique = np.unique(data)
        print(f"{os.path.basename(f)}: {unique}")
    except Exception as e:
        print(f"Error reading {f}: {e}")
