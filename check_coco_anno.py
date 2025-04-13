import json
import os
import argparse
from collections import defaultdict

class COCOAnnotationChecker:
    def __init__(self, annotation_file):
        self.annotation_file = annotation_file
        self.load_annotations()
        
    def load_annotations(self):
        """Load COCO annotation file"""
        print(f"Loading annotations from {self.annotation_file}...")
        with open(self.annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create image_id to image info mapping
        self.images = {img['id']: img for img in self.coco_data['images']}
        
        print(f"Found {len(self.coco_data['images'])} images and {len(self.coco_data['annotations'])} annotations")
    
    def validate_bbox(self, bbox, image_id):
        """Validate if bbox coordinates are valid"""
        x, y, w, h = bbox  # COCO format is [x, y, width, height]
        img_info = self.images[image_id]
        img_width = img_info['width']
        img_height = img_info['height']
        
        errors = []
        
        # Check if coordinates or dimensions are negative
        if x < 0 or y < 0:
            errors.append(f"Negative coordinates found: ({x}, {y})")
        if w <= 0 or h <= 0:
            errors.append(f"Invalid dimensions: width={w}, height={h}")
        
        # Check if bbox extends beyond image dimensions
        if x >= img_width or x + w > img_width:
            errors.append(f"X-coordinates ({x}, {x+w}) out of image width ({img_width})")
        if y >= img_height or y + h > img_height:
            errors.append(f"Y-coordinates ({y}, {y+h}) out of image height ({img_height})")
        
        # Check if bbox area is too small
        if w * h < 1:
            errors.append(f"Bounding box area too small: {w * h} pixels")
            
        return errors
    
    def check_annotations(self):
        """Check all annotations for potential errors"""
        errors_by_image = defaultdict(list)
        total_errors = 0
        
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            bbox = ann['bbox']
            category_id = ann['category_id']
            
            # Get category name
            category_name = next((cat['name'] for cat in self.coco_data['categories'] 
                               if cat['id'] == category_id), 'unknown')
            
            errors = self.validate_bbox(bbox, image_id)
            if errors:
                total_errors += 1
                img_info = self.images[image_id]
                error_info = {
                    'category': category_name,
                    'bbox': bbox,
                    'errors': errors
                }
                errors_by_image[img_info['file_name']].append(error_info)
        
        return errors_by_image, total_errors
    
    def print_error_report(self, errors_by_image, total_errors):
        """Print a detailed error report"""
        print("\n=== COCO Annotation Error Report ===")
        print(f"Total annotations with errors: {total_errors}")
        print(f"Number of images with errors: {len(errors_by_image)}")
        
        if total_errors > 0:
            print("\nDetailed Error Report:")
            for img_name, errors in errors_by_image.items():
                print(f"\nImage: {img_name}")
                for i, error_info in enumerate(errors, 1):
                    print(f"  Object {i} ({error_info['category']}):")
                    print(f"    Bbox: {error_info['bbox']}")
                    for error in error_info['errors']:
                        print(f"    - {error}")

def main():
    parser = argparse.ArgumentParser(description='COCO Annotation Checker')
    parser.add_argument('--train', action='store_true',
                      help='Check train2014 annotations')
    parser.add_argument('--val', action='store_true',
                      help='Check val2014 annotations')
    args = parser.parse_args()
    
    if not (args.train or args.val):
        print("Please specify at least one of --train or --val")
        return
    
    annotation_dir = 'annotations'
    
    if args.train:
        print("\nChecking training set annotations...")
        checker = COCOAnnotationChecker(os.path.join(annotation_dir, 'instances_train2014.json'))
        errors_by_image, total_errors = checker.check_annotations()
        checker.print_error_report(errors_by_image, total_errors)
    
    if args.val:
        print("\nChecking validation set annotations...")
        checker = COCOAnnotationChecker(os.path.join(annotation_dir, 'instances_val2014.json'))
        errors_by_image, total_errors = checker.check_annotations()
        checker.print_error_report(errors_by_image, total_errors)

if __name__ == "__main__":
    main()