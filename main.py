"""
YOLOv8 Training Script for Drone-Balloon Detection Dataset
This script trains a YOLOv8m model on the drone-balloon dataset.
"""

from ultralytics import YOLO
import os

def main():
    """
    Main training function for YOLOv8m model.
    """
    # Initialize YOLOv8m model (medium size)
    model = YOLO('yolov8m.pt')
    
    # Path to dataset configuration file
    data_yaml = 'data.yaml'
    
    # Check if data.yaml exists
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset configuration file '{data_yaml}' not found!")
    
    # Training parameters
    results = model.train(
        data=data_yaml,           # Path to dataset configuration
        epochs=100,               # Number of training epochs
        imgsz=640,                # Image size for training
        batch=16,                 # Batch size (adjust based on GPU memory)
        name='drone_balloon_yolov8m',  # Project name
        patience=50,              # Early stopping patience
        save=True,                # Save checkpoints
        save_period=10,           # Save checkpoint every N epochs
        val=True,                 # Validate during training
        plots=True,               # Generate training plots
        device=0,                 # GPU device (0 for first GPU, 'cpu' for CPU)
        workers=8,                # Number of worker threads for data loading
        project='runs/detect',    # Project directory
    )
    
    # Print training results summary
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f"Best model saved at: {results.save_dir}")
    print("="*50)

if __name__ == '__main__':
    main()

