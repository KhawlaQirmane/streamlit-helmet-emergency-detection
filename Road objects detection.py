from ultralytics import YOLO
import torch
import os
import datetime

# Step 0: Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

# Step 1: Load pre-trained YOLOv8 model
model = YOLO('yolo11n.pt')

# Step 2: Set up training parameters
data_path = r"C:\Users\user\Desktop\yolo_yolo\data.yaml"
project_path = r"C:\Users\user\Desktop\yolo_yolo"

# Ensure a unique folder name to avoid overwriting previous runs
# We'll use the current date and time to make it unique
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
custom_model_name = f"custom_yolov8_{timestamp}_khawla_final_model"

# Step 3: Train the model
results = model.train(
    data=data_path,
    epochs=10,
    batch=16,
    imgsz=640,
    name=custom_model_name,  
    workers=8,
    project=project_path  # Directory where the results folder will be created
)

# Step 4: Explicitly locate and save the best model weights
weights_path = os.path.join(project_path, custom_model_name, 'weights', 'best.pt')
custom_save_path = os.path.join(project_path, f"{custom_model_name}_final.pt")

# Make sure the weights exist before trying to save
if os.path.exists(weights_path):
    # Copy the best weights to a specific path
    import shutil
    shutil.copyfile(weights_path, custom_save_path)
    print(f"Model successfully saved at: {custom_save_path}")
else:
    print(f"Error: Weights file not found at {weights_path}. Please check if training was successful.")

# Step 5: Load the custom-trained model
saved_model = YOLO(custom_save_path)
