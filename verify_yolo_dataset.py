import os
import cv2
import numpy as np

def read_classes(classes_file):
    with open(classes_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

def convert_yolo_to_bbox(image_width, image_height, yolo_bbox):
    # Convert YOLO bbox (x_center, y_center, width, height) to 
    # CV2 bbox (x_min, y_min, x_max, y_max)
    x_center, y_center, width, height = yolo_bbox
    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height
    
    x_min = int(x_center - width/2)
    y_min = int(y_center - height/2)
    x_max = int(x_center + width/2)
    y_max = int(y_center + height/2)
    
    return x_min, y_min, x_max, y_max

def visualize_dataset(dataset_path, dataset_type='train'):
    # Load class names
    classes = read_classes(os.path.join(dataset_path, 'classes.txt'))
    
    # Get all images in the dataset
    images_dir = os.path.join(dataset_path, 'images', dataset_type)
    labels_dir = os.path.join(dataset_path, 'labels', dataset_type)
    
    image_files = sorted(os.listdir(images_dir))
    
    for img_file in image_files:
        # Read image
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:  # Handle symlinks
            real_path = os.path.realpath(img_path)
            img = cv2.imread(real_path)
            if img is None:
                print(f"Could not read image: {img_file}")
                continue
                
        height, width = img.shape[:2]
        
        # Read corresponding label file
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            # Draw each bounding box
            for line in lines:
                class_id, x_center, y_center, w, h = map(float, line.strip().split())
                x_min, y_min, x_max, y_max = convert_yolo_to_bbox(width, height, 
                                                                 (x_center, y_center, w, h))
                
                # Draw rectangle and label
                color = (0, 255, 0)  # Green color for bbox
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                
                # Add class label
                class_name = classes[int(class_id)]
                label = f'{class_name}'
                cv2.putText(img, label, (x_min, y_min-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Show image
        cv2.imshow('YOLO Dataset Verification', img)
        
        # Wait for key press
        key = cv2.waitKey(0)
        if key == ord('q'):  # Press 'q' to quit
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    dataset_path = 'yolov3_dataset'
    print("Showing training images (press any key for next, 'q' to quit)...")
    visualize_dataset(dataset_path, 'train')
    print("Showing validation images...")
    visualize_dataset(dataset_path, 'val')