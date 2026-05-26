import os

root_dir = r"C:\Users\zhuqh\Desktop\sam3\datasets\心脏超声数据集\52.Unity Imaging Colloborative\png-cache"

print(f"Scanning {root_dir}...")
count = 0
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(os.path.join(root, file))
            count += 1
            if count >= 20:
                break
    if count >= 20:
        break

if count == 0:
    print("No images found.")
