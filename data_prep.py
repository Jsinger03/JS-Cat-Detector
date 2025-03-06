# Filename: data_prep.py
import os
import urllib.request
import tarfile
import shutil
from PIL import Image
import numpy as np
import random

# Directories for dataset
BASE_DIR = os.path.join(os.getcwd(), "data")
IMG_DIR = os.path.join(BASE_DIR, "images")
LABEL_DIR = os.path.join(BASE_DIR, "labels")
TRAIN_IMG_DIR = os.path.join(IMG_DIR, "train")
VAL_IMG_DIR = os.path.join(IMG_DIR, "val")
TRAIN_LABEL_DIR = os.path.join(LABEL_DIR, "train")
VAL_LABEL_DIR = os.path.join(LABEL_DIR, "val")

# URLs for the Oxford-IIIT Pet dataset
IMAGES_URL = "http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTATIONS_URL = "http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

# Create necessary directories
for d in [BASE_DIR, IMG_DIR, LABEL_DIR, TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LABEL_DIR, VAL_LABEL_DIR]:
    os.makedirs(d, exist_ok=True)

print("Downloading Oxford-IIIT Pet dataset (images)...")
urllib.request.urlretrieve(IMAGES_URL, filename="images.tar.gz")
print("Downloading Oxford-IIIT Pet dataset (annotations)...")
urllib.request.urlretrieve(ANNOTATIONS_URL, filename="annotations.tar.gz")

# Extract images and annotations
print("Extracting images...")
with tarfile.open("images.tar.gz") as tar:
    tar.extractall(path=BASE_DIR)
print("Extracting annotations...")
with tarfile.open("annotations.tar.gz") as tar:
    tar.extractall(path=BASE_DIR)

# Create a temporary directory for initial extraction
TEMP_IMG_DIR = os.path.join(BASE_DIR, "images_temp")
os.makedirs(TEMP_IMG_DIR, exist_ok=True)

# Move all images to temporary directory
for f in os.listdir(os.path.join(BASE_DIR, "images")):
    if f.endswith(".jpg"):
        src = os.path.join(BASE_DIR, "images", f)
        dst = os.path.join(TEMP_IMG_DIR, f)
        shutil.move(src, dst)

# Clean up any remaining files in the images directory
shutil.rmtree(os.path.join(BASE_DIR, "images"))
os.makedirs(os.path.join(BASE_DIR, "images"))
os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(VAL_IMG_DIR, exist_ok=True)

# List all image files from temporary directory
all_images = [f for f in os.listdir(TEMP_IMG_DIR) if f.endswith(".jpg")]

# Read the list of class labels (to identify cats vs dogs)
classes_file = os.path.join(BASE_DIR, "annotations", "list.txt")
cat_images = []
with open(classes_file, "r") as cf:
    lines = cf.readlines()[6:]  # first 6 lines are header info
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            image_name = parts[0] + ".jpg"
            class_id = int(parts[1])
            species = int(parts[2])  # 1 = cat, 2 = dog (as per dataset documentation)
            if species == 1:  # it's a cat
                cat_images.append(image_name)

# Shuffle and split cat images into train/val (90% train, 10% val)
random.shuffle(cat_images)
split_idx = int(0.9 * len(cat_images))
train_images = cat_images[:split_idx]
val_images = cat_images[split_idx:]

def mask_to_yolo(bb, img_w, img_h):
    """Convert bounding box (xmin, ymin, xmax, ymax) to YOLO format (class xc yc w h)."""
    (xmin, ymin, xmax, ymax) = bb
    # Clamp bounds to image size
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(img_w-1, xmax), min(img_h-1, ymax)
    bw = xmax - xmin + 1
    bh = ymax - ymin + 1
    # YOLO normalized center and size
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    width = bw / float(img_w)
    height = bh / float(img_h)
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

# Process each image in train and val sets
print("Converting annotations to YOLO format...")
for subset, img_list in [("train", train_images), ("val", val_images)]:
    for img_name in img_list:
        src_img_path = os.path.join(TEMP_IMG_DIR, img_name)
        # corresponding segmentation mask (PNG) is in annotations/trimaps with same base name
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(BASE_DIR, "annotations", "trimaps", mask_name)
        if not os.path.exists(src_img_path) or not os.path.exists(mask_path):
            continue  # skip if any file missing
        # Copy image to appropriate folder
        dst_img_path = os.path.join(BASE_DIR, "images", subset, img_name)
        shutil.copy(src_img_path, dst_img_path)
        # Open mask to find bounding box
        mask = Image.open(mask_path)
        mask_np = np.array(mask)
        # In trimap masks, pixel values: 1=foreground, 2=border, 3=background (or similar).
        # We identify foreground (pet) pixels:
        pet_pixels = np.where(mask_np == 1)
        if pet_pixels[0].size == 0:
            # If class labeling is different (some versions use 1,2 as pet classes and 3 as background)
            pet_pixels = np.where(mask_np != 0)
        if pet_pixels[0].size == 0:
            continue  # no pet found (shouldn't happen for labeled images)
        ymin, ymax = pet_pixels[0].min(), pet_pixels[0].max()
        xmin, xmax = pet_pixels[1].min(), pet_pixels[1].max()
        img = Image.open(src_img_path)
        img_w, img_h = img.size
        yolo_line = mask_to_yolo((xmin, ymin, xmax, ymax), img_w, img_h)
        # Write label file
        label_file = os.path.join(BASE_DIR, "labels", subset, os.path.splitext(img_name)[0] + ".txt")
        with open(label_file, "w") as lf:
            lf.write(yolo_line)
print("Dataset prepared. Train images:", len(train_images), "Val images:", len(val_images))

# Create dataset YAML configuration for YOLO training
dataset_yaml = os.path.join(BASE_DIR, "pet_cat.yaml")
with open(dataset_yaml, "w") as yf:
    yf.write(f"path: {BASE_DIR}\n")
    yf.write("train: images/train\n")
    yf.write("val: images/val\n")
    yf.write("names: [\'cat\']\n")
    yf.write("nc: 1\n")  # number of classes
print("Created dataset config:", dataset_yaml)

# Cleanup temporary directory
shutil.rmtree(TEMP_IMG_DIR)