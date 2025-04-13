import cv2
import os
import xml.etree.ElementTree as ET
import numpy as np
import argparse

class PascalVOCViewer:
    def __init__(self, pascal_dir, auto_mode=False):
        self.pascal_dir = pascal_dir
        self.images_dir = os.path.join(pascal_dir, 'images')
        self.labels_dir = os.path.join(pascal_dir, 'labels')
        self.auto_mode = auto_mode
        
        # Read train.txt to get the list of images
        with open(os.path.join(pascal_dir, 'train.txt'), 'r') as f:
            self.image_list = f.read().strip().split('\n')
        
        self.current_idx = 0
        self.window_name = 'Pascal VOC Annotation Viewer'
        
        # Create colors for different classes
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

    def validate_bbox(self, bbox, image_shape):
        """Validate bounding box coordinates"""
        xmin, ymin, xmax, ymax = bbox
        height, width = image_shape[:2]
        
        errors = []
        
        # Check if coordinates are non-negative
        if xmin < 0 or ymin < 0:
            errors.append(f"Negative coordinates found: ({xmin}, {ymin})")
        
        # Check if max > min
        if xmax <= xmin:
            errors.append(f"Invalid x-coordinates: xmax ({xmax}) <= xmin ({xmin})")
        if ymax <= ymin:
            errors.append(f"Invalid y-coordinates: ymax ({ymax}) <= ymin ({ymin})")
        
        # Check if coordinates are within image dimensions
        if xmin >= width or xmax > width:
            errors.append(f"X-coordinates ({xmin}, {xmax}) out of image width ({width})")
        if ymin >= height or ymax > height:
            errors.append(f"Y-coordinates ({ymin}, {ymax}) out of image height ({height})")
               
        return errors

    def read_xml_annotation(self, xml_path):
        """Read Pascal VOC XML annotation file"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
            objects.append({
                'name': name,
                'bbox': [xmin, ymin, xmax, ymax]
            })
        return objects

    def draw_annotations(self, image, objects, validation_errors=None):
        """Draw bounding boxes and labels on the image"""
        for i, obj in enumerate(objects):
            color = tuple(map(int, self.colors[i % len(self.colors)]))
            if validation_errors and i in validation_errors:
                color = (0, 0, 255)  # Red color for invalid boxes
            bbox = obj['bbox']
            name = obj['name']
            
            # Draw rectangle
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label background
            text = f"{name}" if i not in (validation_errors or {}) else f"{name} (ERROR)"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(image, (bbox[0], bbox[1] - text_size[1] - 4),
                         (bbox[0] + text_size[0], bbox[1]), color, -1)
            
            # Draw text
            cv2.putText(image, text, (bbox[0], bbox[1] - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return image

    def show_current_image(self):
        """Show current image with annotations"""
        if 0 <= self.current_idx < len(self.image_list):
            image_name = self.image_list[self.current_idx]
            img_path = os.path.join(self.images_dir, f'{image_name}.jpg')
            xml_path = os.path.join(self.labels_dir, f'{image_name}.xml')
            
            # Read image and annotations
            image = cv2.imread(img_path)
            if image is None:
                print(f"Error: Could not read image {img_path}")
                return False
            
            objects = self.read_xml_annotation(xml_path)
            
            # Validate annotations
            validation_errors = {}
            has_errors = False
            for i, obj in enumerate(objects):
                errors = self.validate_bbox(obj['bbox'], image.shape)
                if errors:
                    has_errors = True
                    validation_errors[i] = errors
                    print(f"\nValidation errors in object {i+1} ({obj['name']}):")
                    for error in errors:
                        print(f"  - {error}")
            
            # Draw annotations on image
            annotated_image = self.draw_annotations(image.copy(), objects, validation_errors if has_errors else None)
            
            # Show image info
            print(f"\nImage {self.current_idx + 1}/{len(self.image_list)}: {image_name}")
            print(f"Number of objects: {len(objects)}")
            
            # Show image
            cv2.imshow(self.window_name, annotated_image)
            
            # In auto mode, return whether we should pause
            return has_errors
            
        return False

    def run(self):
        """Main viewing loop"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        print("Controls:")
        if not self.auto_mode:
            print("'n' - next image")
            print("'p' - previous image")
        else:
            print("Auto mode: Pausing only on validation errors")
            print("'n' - continue to next image when paused")
            print("'p' - previous image when paused")
        print("'q' - quit")
        
        should_pause = self.show_current_image()
        
        while True:
            if self.auto_mode and not should_pause:
                key = cv2.waitKey(1000) & 0xFF  # Wait 1 second before advancing
                if key == ord('q'):
                    break
                self.current_idx = min(self.current_idx + 1, len(self.image_list) - 1)
                should_pause = self.show_current_image()
            else:
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('n'):  # Next image
                    self.current_idx = min(self.current_idx + 1, len(self.image_list) - 1)
                    should_pause = self.show_current_image()
                elif key == ord('p'):  # Previous image
                    self.current_idx = max(self.current_idx - 1, 0)
                    should_pause = self.show_current_image()
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pascal VOC Annotation Viewer')
    parser.add_argument('-a', '--auto', action='store_true',
                      help='Enable auto-advance mode (pause only on validation errors)')
    args = parser.parse_args()
    
    viewer = PascalVOCViewer('pascal_coco', auto_mode=args.auto)
    viewer.run()