# Object Detection with OpenCV and YOLO 👓

This project demonstrates object detection using OpenCV and the YOLO (You Only Look Once) deep learning model.  It takes an image as input and identifies various objects within the image, drawing bounding boxes and labels around them.

## Requirements

* Python 3.6+
* OpenCV (`cv2`)
* NumPy (`numpy`)

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/LadyKerr/opencv-obj-detection.git
    cd opencv-obj-detection
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Download YOLOv3 weights and configuration:**

    You'll need to download the YOLOv3 weights (`yolov3.weights`), configuration file (`yolov3.cfg`), and class names file (`coco.names`). Place these files in a directory named `yolo_files` within the project directory.  You can find links to download these files from the official YOLO website or other reputable sources.  Ensure the directory structure looks like this:

    ```
    opencv-obj-detection/
    ├── yolo_files/
    │   ├── yolov3.weights
    │   ├── yolov3.cfg
    │   └── coco.names
    └── ... other project files
    ```

    1. Download YOLOv3 weights file (237 MB):
   ```bash
   wget https://pjreddie.com/media/files/yolov3.weights
   ```

   2. Download YOLOv3 configuration file:
   ```bash
    wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
    ```

    3. Download the COCO class names file:
    ```bash
    wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
    ```
Place these files in the project root directory before running the detection script.

## Usage

To run the object detection script:

```bash
python detection.py <path_to_image> [--output <path_to_output_image>]
```

* `<path_to_image>`:  The path to the input image file.
* `--output <path_to_output_image>` (optional): The path to save the output image with detected objects. If not provided, the result will be displayed in a window.

**Example:**

```bash
python detection.py images/input.jpg --output images/output.jpg
```

This will process `images/input.jpg` and save the result to `images/output.jpg`.

## Error Handling

The script includes error handling for common issues such as:

* **File not found:** If the input image file doesn't exist.
* **Image loading errors:** If the image file is corrupted or cannot be loaded.
* **YOLO model loading errors:** If there's an issue loading the YOLO model files.

Specific error messages will be printed to the console if any of these errors occur.


## Contributing

Contributions are welcome!  Please feel free to submit pull requests.

## License

This project is licensed under the [MIT License](LICENSE).