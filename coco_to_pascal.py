import os
import json
from xml.etree.ElementTree import Element, SubElement, ElementTree
import shutil

def create_pascal_voc_dirs(output_dir):
    """Create Pascal VOC directory structure"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

def validate_and_fix_bbox(bbox, img_width, img_height, min_size=1, expansion_factor=1.2):
    """Validate and attempt to fix COCO format bbox [x, y, width, height]"""
    x, y, w, h = bbox
    
    # Check if bbox is valid
    is_valid = True
    needs_fixing = False
    
    # Check for negative coordinates
    if x < 0 or y < 0:
        is_valid = False
    
    # Check for zero or negative dimensions
    if w <= 0 or h <= 0:
        needs_fixing = True  # Try to fix zero dimensions instead of rejecting
    
    # Check for out-of-bounds coordinates
    if x >= img_width or y >= img_height:
        is_valid = False
    
    # Check if bbox is too small
    if w * h < min_size:
        needs_fixing = True
    
    if not is_valid:
        return None, False  # Cannot fix invalid boxes
        
    if needs_fixing:
        # For zero-width or zero-height boxes, expand by at least 2 pixels
        min_expansion = 2  # Minimum expansion in pixels
        
        # Calculate expanded dimensions ensuring minimum size
        if w <= 0:
            w = min_expansion
        if h <= 0:
            h = min_expansion
            
        # Further expand if still too small
        if w * h < min_size:
            scale = (min_size / (w * h)) ** 0.5 * expansion_factor
            w *= scale
            h *= scale
        
        # Ensure the box stays within image bounds
        x = max(0, min(x, img_width - w))
        y = max(0, min(y, img_height - h))
        
        return [x, y, w, h], True
        
    # Box is valid and doesn't need fixing
    return bbox, False

def create_xml_annotation(img_data, annotations, categories, img_path):
    """Create Pascal VOC XML annotation"""
    root = Element('annotation')
    
    # Add basic image information
    folder = SubElement(root, 'folder')
    folder.text = 'images'
    
    filename = SubElement(root, 'filename')
    filename.text = img_data['file_name']
    
    path = SubElement(root, 'path')
    path.text = img_path
    
    source = SubElement(root, 'source')
    database = SubElement(source, 'database')
    database.text = 'COCO2014'
    
    # Image size information
    size = SubElement(root, 'size')
    width = SubElement(size, 'width')
    width.text = str(img_data['width'])
    height = SubElement(size, 'height')
    height.text = str(img_data['height'])
    depth = SubElement(size, 'depth')
    depth.text = '3'
    
    # Add object annotations
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
    
    filtered_count = 0
    fixed_count = 0
    
    filtered_annotations = []
    for ann in annotations:
        bbox = ann['bbox']
        fixed_bbox, was_fixed = validate_and_fix_bbox(bbox, img_data['width'], img_data['height'])
        
        if fixed_bbox is None:
            filtered_count += 1
            continue
            
        if was_fixed:
            fixed_count += 1
            bbox = fixed_bbox
            
        # Convert COCO bbox [x,y,w,h] to Pascal VOC [xmin,ymin,xmax,ymax]
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[0] + max(1, bbox[2]))  # Ensure at least 1 pixel width
        ymax = int(bbox[1] + max(1, bbox[3]))  # Ensure at least 1 pixel height
        
        # Additional validation in Pascal VOC format
        if xmax <= xmin:
            xmax = xmin + 1  # Ensure at least 1 pixel width
        if ymax <= ymin:
            ymax = ymin + 1  # Ensure at least 1 pixel height
            
        filtered_annotations.append((ann['category_id'], [xmin, ymin, xmax, ymax]))
        
        obj = SubElement(root, 'object')
        name = SubElement(obj, 'name')
        name.text = cat_id_to_name[ann['category_id']]
        
        bndbox = SubElement(obj, 'bndbox')
        xmin_elem = SubElement(bndbox, 'xmin')
        xmin_elem.text = str(xmin)
        ymin_elem = SubElement(bndbox, 'ymin')
        ymin_elem.text = str(ymin)
        xmax_elem = SubElement(bndbox, 'xmax')
        xmax_elem.text = str(xmax)
        ymax_elem = SubElement(bndbox, 'ymax')
        ymax_elem.text = str(ymax)
    
    return root, filtered_count, fixed_count, len(filtered_annotations) > 0

def convert_coco_to_pascal(coco_path, img_dir, output_dir, dataset_type):
    """Convert COCO format to Pascal VOC format"""
    # Create output directories if they don't exist
    create_pascal_voc_dirs(output_dir)
    
    # Read COCO annotations
    with open(coco_path, 'r') as f:
        coco = json.load(f)
    
    # Extract and save class names to classes.txt (only for first dataset)
    if not os.path.exists(os.path.join(output_dir, 'classes.txt')):
        class_names = [cat['name'] for cat in coco['categories']]
        with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
            f.write('\n'.join(class_names))
    
    # Create image id to annotations mapping
    img_to_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # Statistics
    total_filtered = 0
    total_fixed = 0
    total_images = 0
    total_valid_images = 0
    
    # Create dataset-specific .txt file (train.txt or val.txt)
    dataset_txt = []
    
    # Process each image
    for img in coco['images']:
        img_id = img['id']
        img_name = img['file_name']
        img_path = os.path.join(img_dir, img_name)
        
        total_images += 1
        
        # Skip if image file doesn't exist
        if not os.path.exists(img_path):
            continue
        
        # Skip if image has no annotations
        if img_id not in img_to_anns:
            continue
            
        # Create XML annotation and get statistics
        xml_root, filtered_count, fixed_count, has_valid_annotations = create_xml_annotation(
            img, img_to_anns[img_id], coco['categories'], img_path
        )
        
        # Skip if no valid annotations remain
        if not has_valid_annotations:
            continue
            
        total_filtered += filtered_count
        total_fixed += fixed_count
        total_valid_images += 1
        
        # Create symbolic link for image
        src_img_path = os.path.abspath(img_path)
        dst_img_path = os.path.join(output_dir, 'images', img_name)
        if os.path.exists(dst_img_path):
            os.remove(dst_img_path)
        os.symlink(src_img_path, dst_img_path)
        
        # Add to dataset txt file
        dataset_txt.append(os.path.splitext(img_name)[0])
        
        # Save XML file
        xml_path = os.path.join(output_dir, 'labels', 
                              os.path.splitext(img_name)[0] + '.xml')
        tree = ElementTree(xml_root)
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    
    # Write dataset txt file (train.txt or val.txt)
    with open(os.path.join(output_dir, f'{dataset_type}.txt'), 'w') as f:
        f.write('\n'.join(dataset_txt))
        
    # Print statistics
    print(f"\nConversion Statistics for {dataset_type}:")
    print(f"Total images processed: {total_images}")
    print(f"Images with valid annotations: {total_valid_images}")
    print(f"Total annotations filtered out: {total_filtered}")
    print(f"Total annotations fixed: {total_fixed}")

if __name__ == "__main__":
    print("Starting conversion to Pascal VOC format...")
    output_dir = 'pascal_coco'
    
    # Convert training dataset
    print("\nConverting training dataset...")
    convert_coco_to_pascal(
        'annotations/instances_train2014.json',
        'train2014',
        output_dir,
        'train'
    )
    
    # Convert validation dataset
    print("\nConverting validation dataset...")
    convert_coco_to_pascal(
        'annotations/instances_val2014.json',
        'val2014',
        output_dir,
        'val'
    )
    
    print("Conversion complete!")
