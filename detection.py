import cv2
import numpy as np
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Object Detection')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--output', '-o', help='Path to output image')
    return parser.parse_args()

def load_yolo():
    net = cv2.dnn.readNet(
        "yolo_files/yolov3.weights",
        "yolo_files/yolov3.cfg"
    )
    with open("yolo_files/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    return net, classes

def detect_objects(image, net, classes):
    height, width = image.shape[:2]
    
    # Create blob from image
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Run forward pass
    outputs = net.forward(output_layers)
    
    # Process detections
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    return boxes, confidences, class_ids

def draw_detections(image, boxes, confidences, class_ids, classes):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = colors[class_ids[i]]
        
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

def main():
    args = parse_args()
    image_path = Path(args.image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Load YOLO model
    try:
        net, classes = load_yolo()
    except Exception as e:
        raise RuntimeError(f"Error loading YOLO model: {e}")
    
    # Detect objects
    boxes, confidences, class_ids = detect_objects(image, net, classes)
    
    # Draw results
    result = draw_detections(image, boxes, confidences, class_ids, classes)
    
    # Save or display result
    if args.output:
        cv2.imwrite(args.output, result)
    else:
        cv2.imshow("Object Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)