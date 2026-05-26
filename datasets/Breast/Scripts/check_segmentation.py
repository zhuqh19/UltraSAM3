import os
import json
import glob
import numpy as np

DATASETS_ROOT = r"C:\Users\zhuqh\Desktop\sam3\datasets\乳腺超声数据集\Breast\Datasets"

def is_box_segmentation(segmentation):
    # Check if segmentation is just a box
    # segmentation is usually [[x1, y1, x2, y2, ...]]
    if not segmentation:
        return False
    
    points = segmentation[0]
    if len(points) != 8: # 4 points * 2 coordinates
        return False
    
    # Extract x and y
    xs = points[0::2]
    ys = points[1::2]
    
    # Check if it forms a rectangle aligned with axes
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    # Calculate area of bbox
    bbox_area = (x_max - x_min) * (y_max - y_min)
    
    # Calculate polygon area (Shoelace formula) or just check coords
    # If it's a rectangle, the points should be (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax) in some order
    
    unique_xs = set(xs)
    unique_ys = set(ys)
    
    return len(unique_xs) <= 2 and len(unique_ys) <= 2

def check_dataset(dataset_name):
    dataset_path = os.path.join(DATASETS_ROOT, dataset_name)
    json_files = glob.glob(os.path.join(dataset_path, "*", "_annotations.coco.json"))
    
    if not json_files:
        return "No JSON found"
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            annotations = data.get('annotations', [])
            if not annotations:
                return "Empty annotations"
            
            # Check first few annotations
            sample_anns = annotations[:10]
            segmentation_counts = 0
            box_segmentation_counts = 0
            
            for ann in sample_anns:
                seg = ann.get('segmentation', [])
                if seg:
                    segmentation_counts += 1
                    if is_box_segmentation(seg):
                        box_segmentation_counts += 1
            
            if segmentation_counts == 0:
                return "No segmentation field"
            elif box_segmentation_counts == segmentation_counts:
                return "Only BBox (converted to box-mask)"
            else:
                return "Has Real Segmentation"
                
        except Exception as e:
            return f"Error: {e}"
            
    return "Unknown"

def main():
    datasets = [d for d in os.listdir(DATASETS_ROOT) if os.path.isdir(os.path.join(DATASETS_ROOT, d))]
    
    print(f"{'Dataset':<25} | {'Status':<30}")
    print("-" * 60)
    
    for ds in datasets:
        status = check_dataset(ds)
        print(f"{ds:<25} | {status:<30}")

if __name__ == "__main__":
    main()
