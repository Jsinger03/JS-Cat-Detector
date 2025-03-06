import os
import shutil

# Copy the best weights to a convenient location (models/cat_detector.pt)
best_weights = os.path.join("models", "cat_yolo_run2", "weights", "best.pt")
if os.path.exists(best_weights):
    shutil.copy(best_weights, os.path.join("models", "cat_detector.pt"))
    print("Best model copied to models/cat_detector.pt")
else:
    print("Best model not found in cat_yolo_run2/weights/") 