import json

json_path = r"C:\Users\zhuqh\Desktop\sam3\datasets\心脏超声数据集\52.Unity Imaging Colloborative\u4s-labels\labels-all.json"

try:
    with open(json_path, "r") as f:
        data = json.load(f)
    
    found = False
    for k, v in data.items():
        labels = v.get("labels", {})
        if "curve-lv-endo" in labels:
            print(f"Found curve-lv-endo in {k}")
            print(json.dumps(labels["curve-lv-endo"], indent=2))
            found = True
            break
            
    if not found:
        print("curve-lv-endo not found in any image.")

except Exception as e:
    print(f"Error: {e}")
