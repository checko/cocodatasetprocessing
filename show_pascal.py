import cv2
import os
import xml.etree.ElementTree as ET
import numpy as np

class PascalVOCViewer:
    def __init__(self, pascal_dir):
        self.pascal_dir = pascal_dir
        self.images_dir = os.path.join(pascal_dir, 'images')
        self.labels_dir = os.path.join(pascal_dir, 'labels')
        
        # Read train.txt to get the list of images
        with open(os.path.join(pascal_dir, 'train.txt'), 'r') as f:
            self.image_list = f.read().strip().split('\n')
        
        self.current_idx = 0
        self.window_name = 'Pascal VOC Annotation Viewer'
        
        # Create colors for different classes
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

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

    def draw_annotations(self, image, objects):
        """Draw bounding boxes and labels on the image"""
        for i, obj in enumerate(objects):
            color = tuple(map(int, self.colors[i % len(self.colors)]))
            bbox = obj['bbox']
            name = obj['name']
            
            # Draw rectangle
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label background
            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(image, (bbox[0], bbox[1] - text_size[1] - 4),
                         (bbox[0] + text_size[0], bbox[1]), color, -1)
            
            # Draw text
            cv2.putText(image, name, (bbox[0], bbox[1] - 2),
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
                return
            
            objects = self.read_xml_annotation(xml_path)
            
            # Draw annotations on image
            annotated_image = self.draw_annotations(image.copy(), objects)
            
            # Show image info
            print(f"\nImage {self.current_idx + 1}/{len(self.image_list)}: {image_name}")
            print(f"Number of objects: {len(objects)}")
            
            # Show image
            cv2.imshow(self.window_name, annotated_image)

    def run(self):
        """Main viewing loop"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        print("Controls:")
        print("'n' - next image")
        print("'p' - previous image")
        print("'q' - quit")
        
        self.show_current_image()
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('n'):  # Next image
                self.current_idx = min(self.current_idx + 1, len(self.image_list) - 1)
                self.show_current_image()
            elif key == ord('p'):  # Previous image
                self.current_idx = max(self.current_idx - 1, 0)
                self.show_current_image()
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    viewer = PascalVOCViewer('pascal_coco')
    viewer.run()