# Filename: train_yolo.py
import os
import shutil
from ultralytics import YOLO

DATA_CONFIG = "data/pet_cat.yaml"      # path to dataset config from data_prep.py
PRETRAINED_MODEL = "yolov8n.pt"        # using YOLOv8 nano as a base (pretrained on COCO)
EPOCHS = 20                            # you can adjust epochs as needed
DEVICE = "mps"                         # use Apple Silicon GPU (Metal Performance Shaders)

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

# Load YOLO model
model = YOLO(PRETRAINED_MODEL)  # this will automatically download yolov8n.pt if not present

# Train the model on our data
results = model.train(data=DATA_CONFIG, epochs=EPOCHS, imgsz=640, device=DEVICE, project="models", name="cat_yolo_run")

# The above training will output artifacts in models/cat_yolo_run/, including best.pt
# We will copy the best weights to a convenient location (models/cat_detector.pt)
best_weights = os.path.join("models", "cat_yolo_run", "weights", "best.pt")
if os.path.exists(best_weights):
    shutil.copy(best_weights, os.path.join("models", "cat_detector.pt"))
    print("Training complete. Best model saved to models/cat_detector.pt")
else:
    print("Training complete. Best model not found, check training output.")