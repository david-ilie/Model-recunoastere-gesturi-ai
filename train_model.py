import subprocess
import signal
import time
import sys
from ultralytics import YOLO
import torch

def train_model():
    try:

        if not torch.cuda.is_available():
            print("CUDA is not available. Please check your GPU and CUDA installation.")
            return False
            
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        

        model = YOLO('yolov5su.pt')
        

        results = model.train(
            data='gesture_data.yaml',
            epochs=30,
            imgsz=640,
            batch=16,
            workers=4,
            device=0,
            patience=50,
            save=True,
            save_period=5,
            verbose=True
        )
        
        return True
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Attempting to stop gracefully...")
        return False
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting training...")
    success = train_model()
    if not success:
        print("Training did not complete successfully.")
    else:
        print("Training completed successfully.") 