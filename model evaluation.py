from ultralytics import YOLO

# Load the trained model
model = YOLO(r"C:\Users\user\Desktop\yolo_yolo\custom_yolov8_20250505_125324_yahya_final_model_final.pt")

# Evaluate the model on test data
metrics = model.val(
    data= r"C:\Users\user\Desktop\yolo_yolo\data.yaml",  # Path to the dataset configuration file
    split='test',      # Use test split
    imgsz=640,         # Input size (same as training)
    conf=0.5,         # Confidence threshold
    iou=0.5            # IoU threshold
)

print(metrics)  # Display performance metrics
