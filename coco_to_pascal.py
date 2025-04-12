import os
import json
from xml.etree.ElementTree import Element, SubElement, ElementTree
import shutil

def create_pascal_voc_dirs(output_dir):
    """Create Pascal VOC directory structure"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

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
    
    for ann in annotations:
        obj = SubElement(root, 'object')
        name = SubElement(obj, 'name')
        name.text = cat_id_to_name[ann['category_id']]
        
        bbox = ann['bbox']
        bndbox = SubElement(obj, 'bndbox')
        xmin = SubElement(bndbox, 'xmin')
        xmin.text = str(int(bbox[0]))
        ymin = SubElement(bndbox, 'ymin')
        ymin.text = str(int(bbox[1]))
        xmax = SubElement(bndbox, 'xmax')
        xmax.text = str(int(bbox[0] + bbox[2]))
        ymax = SubElement(bndbox, 'ymax')
        ymax.text = str(int(bbox[1] + bbox[3]))
        
    return root

def convert_coco_to_pascal(coco_path, img_dir, output_dir):
    """Convert COCO format to Pascal VOC format"""
    # Create output directories
    create_pascal_voc_dirs(output_dir)
    
    # Read COCO annotations
    with open(coco_path, 'r') as f:
        coco = json.load(f)
    
    # Extract and save class names to classes.txt
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
    
    # Create train.txt
    train_txt = []
    
    # Process each image
    for img in coco['images']:
        img_id = img['id']
        img_name = img['file_name']
        img_path = os.path.join(img_dir, img_name)
        
        if not os.path.exists(img_path):
            continue
            
        # Create symbolic link for image
        src_img_path = os.path.abspath(img_path)
        dst_img_path = os.path.join(output_dir, 'images', img_name)
        if os.path.exists(dst_img_path):
            os.remove(dst_img_path)
        os.symlink(src_img_path, dst_img_path)
        
        # Add to train.txt
        train_txt.append(os.path.splitext(img_name)[0])
        
        # Create XML annotation if annotations exist
        if img_id in img_to_anns:
            xml_root = create_xml_annotation(img, img_to_anns[img_id], 
                                          coco['categories'], dst_img_path)
            
            # Save XML file
            xml_path = os.path.join(output_dir, 'labels', 
                                  os.path.splitext(img_name)[0] + '.xml')
            tree = ElementTree(xml_root)
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    
    # Write train.txt
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_txt))

if __name__ == "__main__":
    print("Starting conversion to Pascal VOC format...")
    
    # Convert training dataset
    convert_coco_to_pascal(
        'annotations/instances_train2014.json',
        'train2014',
        'pascal_coco'
    )
    
    print("Conversion complete!")
