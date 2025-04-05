import json
import os
from PIL import Image

def convert_bbox_to_yolo(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[2]/2.0) * dw
    y = (box[1] + box[3]/2.0) * dh
    w = box[2] * dw
    h = box[3] * dh
    return (x,y,w,h)

def convert_coco_to_yolo(coco_path, img_dir, output_dir, dataset_type):
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images', dataset_type), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', dataset_type), exist_ok=True)
    
    # Read COCO annotations
    with open(coco_path, 'r') as f:
        coco = json.load(f)
    
    # Create category id to continuous index mapping (YOLO format)
    cat_id_to_cont_id = {cat['id']: i for i, cat in enumerate(coco['categories'])}
    
    # Save category names (only once, not for each dataset type)
    if dataset_type == 'train':
        with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
            for cat in coco['categories']:
                f.write(f"{cat['name']}\n")
    
    # Create image id to annotations mapping
    img_to_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # Process each image
    for img in coco['images']:
        img_id = img['id']
        img_name = img['file_name']
        img_path = os.path.join(img_dir, img_name)
        
        if not os.path.exists(img_path):
            continue
            
        with Image.open(img_path) as img_obj:
            width, height = img_obj.size
        
        if img_id in img_to_anns:
            label_path = os.path.join(output_dir, 'labels', dataset_type,
                                    os.path.splitext(img_name)[0] + '.txt')
            
            with open(label_path, 'w') as f:
                for ann in img_to_anns[img_id]:
                    bbox = ann['bbox']
                    yolo_bbox = convert_bbox_to_yolo((width, height), bbox)
                    category_id = cat_id_to_cont_id[ann['category_id']]
                    f.write(f"{category_id} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n")
        
        os.system(f"cp {img_path} {os.path.join(output_dir, 'images', dataset_type, img_name)}")

print("Starting conversion...")
# Convert training dataset
convert_coco_to_yolo(
    'annotations/instances_train2014.json',
    'train2014',
    'yolov3_dataset',
    'train'
)
# Convert validation dataset
convert_coco_to_yolo(
    'annotations/instances_val2014.json',
    'val2014',
    'yolov3_dataset',
    'val'
)
print("Conversion complete!")
