import json

json_path = r"C:\Users\zhuqh\Desktop\sam3\datasets\心脏超声数据集\52.Unity Imaging Colloborative\u4s-labels\labels-all.json"

try:
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Get the first item
    first_key = list(data.keys())[0]
    print(f"First image: {first_key}")
    print("Keys in first image labels:", list(data[first_key]["labels"].keys()))
    
    # Check if 'lv-endo' or similar exists in any image
    found_lv = False
    for k, v in data.items():
        labels = v.get("labels", {})
        for label_name in labels:
            if "lv" in label_name or "contour" in label_name:
                print(f"Found label: {label_name} in {k}")
                print(json.dumps(labels[label_name], indent=2))
                found_lv = True
                break
        if found_lv:
            break
            
except Exception as e:
    print(f"Error: {e}")
