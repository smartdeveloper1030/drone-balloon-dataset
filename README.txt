================================================================================
YOLOv8 Training for Drone-Balloon Detection Dataset
================================================================================

This project trains a YOLOv8m (medium) model to detect drones and balloons in images.

DATASET STRUCTURE:
------------------
The dataset should be organized as follows:
    drone-balloon-dataset/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    └── valid/
        ├── images/
        └── labels/

INSTALLATION:
-------------
1. Install Python 3.8 or higher
2. Install required libraries:
   pip install -r requirements.txt

   Or install individually:
   pip install ultralytics torch torchvision

TRAINING:
---------
1. Ensure your dataset is properly organized (see DATASET STRUCTURE above)
2. Make sure data.yaml is configured correctly with proper paths
3. Run the training script:
   python main.py

TRAINING PARAMETERS (can be modified in main.py):
-------------------------------------------------
- epochs: 100 (number of training iterations)
- imgsz: 640 (image size for training)
- batch: 16 (batch size - adjust based on GPU memory)
- device: 0 (GPU device number, use 'cpu' for CPU training)
- workers: 8 (number of data loading workers)

OUTPUT:
-------
Training results will be saved in:
    runs/detect/drone_balloon_yolov8m/

This directory contains:
- weights/best.pt: Best model weights
- weights/last.pt: Last checkpoint weights
- results.png: Training metrics visualization
- confusion_matrix.png: Confusion matrix
- Other training artifacts

NOTES:
------
- For CPU training, change device='cpu' in main.py
- Adjust batch size based on available GPU memory
- The model will automatically download yolov8m.pt on first run
- Training progress will be displayed in the console
- Early stopping is enabled (patience=50 epochs)

TROUBLESHOOTING:
---------------
1. If you get CUDA out of memory error:
   - Reduce batch size (e.g., batch=8 or batch=4)
   - Reduce image size (e.g., imgsz=416)

2. If data.yaml paths are incorrect:
   - Update paths in data.yaml to match your directory structure
   - Use absolute paths if relative paths don't work

3. For CPU training:
   - Set device='cpu' in main.py
   - Reduce batch size and workers

================================================================================

