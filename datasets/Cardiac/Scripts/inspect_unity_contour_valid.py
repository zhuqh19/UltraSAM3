import json

json_path = r"C:\Users\zhuqh\Desktop\sam3\datasets\心脏超声数据集\52.Unity Imaging Colloborative\u4s-labels\labels-all.json"

try:
    with open(json_path, "r") as f:
        data = json.load(f)
    
    found = False
    for k, v in data.items():
        labels = v.get("labels", {})
        if "curve-lv-endo" in labels:
            contour = labels["curve-lv-endo"]
            # Check if it has valid points
            valid_points = [p for p in contour if p.get("type") != "off" and p.get("x") != ""]
            if valid_points:
                print(f"Found valid curve-lv-endo in {k}")
                print(f"Number of points: {len(valid_points)}")
                print(json.dumps(valid_points[:5], indent=2))
                found = True
                break
            
    if not found:
        print("No valid curve-lv-endo found.")

except Exception as e:
    print(f"Error: {e}")
