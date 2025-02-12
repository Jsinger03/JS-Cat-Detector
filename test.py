print("Script started...")  # Debug print

import os
print("finished imports")
import cv2
print("finished imports")
import torch
print("finished imports")
from torchvision import transforms, models
print("finished imports")
from torch import nn
print("finished imports")
from torch.utils.data import DataLoader
import signal


# Paths to models
CAT_MODEL_PATH = "models/cat_detection_model.pth"

# Load the trained cat detection model
def load_cat_model():
    print("Checking if model file exists...")
    if not os.path.exists(CAT_MODEL_PATH):
        print(f"Error: Model file {CAT_MODEL_PATH} does not exist.")
        return None

    print("Loading the model architecture...")
    model = models.resnet50(weights=None)  # Load without pretrained weights
    num_features = model.fc.in_features

    # Match the architecture used during training
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )

    print("Loading model weights...")
    model.load_state_dict(torch.load(CAT_MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    print("Model loaded successfully.")
    return model

# Function to handle timeouts
def handler(signum, frame):
    raise Exception("Timeout")

# Function to test if an image contains a cat
def test_image_for_cat(image_path, model):
    if model is None:
        print("Error: Cat model is not loaded.")
        return None

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Move the model and input to the appropriate device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    image_tensor = image_tensor.to(device)

    # Run the model
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.sigmoid(output).item()

    # Interpret the result and print to terminal
    if prediction > 0.5:
        print(f"The image at {image_path} is predicted to contain a cat with confidence {prediction:.2f}.")
    else:
        print(f"The image at {image_path} is predicted not to contain a cat with confidence {1 - prediction:.2f}.")

    return prediction  # Ensure the prediction is returned

# Function to test if images in a batch contain cats
def test_images_for_cats(image_paths, model, device):
    images = []
    for image_path in image_paths:
        try:
            # Set a timeout for reading each image
            signal.signal(signal.SIGALRM, handler)
            signal.setitimer(signal.ITIMER_REAL, 0.1)  # Timeout after 0.1 seconds

            image = cv2.imread(image_path)
            if image is not None:
                preprocess = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                image_tensor = preprocess(image)
                images.append(image_tensor)
            else:
                print(f"Warning: Unable to read image at {image_path}")

            signal.setitimer(signal.ITIMER_REAL, 0)  # Disable the alarm
        except Exception as e:
            #print(f"Error reading image {image_path}: {e}")
            continue

    if not images:
        return []

    try:
        # Stack images into a batch
        image_batch = torch.stack(images).to(device)
        print(f"Loaded batch of {len(images)} images to {device}")

        # Run the model
        model.eval()
        with torch.no_grad():
            outputs = model(image_batch)
            predictions = torch.sigmoid(outputs).squeeze().tolist()

        # Clear the batch from memory
        del image_batch
        #torch.cuda.empty_cache()  # Use this if you are using CUDA, otherwise it's not needed

        return predictions
    except Exception as e:
        print(f"Error processing batch: {e}")
        return []

def test_images_in_subfolders(model):
    subfolders = [
        "data/image_data/CAT_00",
        "data/image_data/CAT_01",
        "data/image_data/CAT_02",
        "data/image_data/CAT_03",
        "data/image_data/CAT_04",
        "data/image_data/CAT_05",
        "data/image_data/CAT_06"
    ]
    
    # Use MPS if available, otherwise fall back to CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print(f"Model moved to {device}")

    batch_size = 32  # Adjust based on GPU memory
    for subfolder_path in subfolders:
        print(f"Processing subfolder: {subfolder_path}")
        num_images = 0
        num_cats = 0
        image_paths = [os.path.join(subfolder_path, file) for file in os.listdir(subfolder_path) if file.endswith(".jpg")]
        
        if not image_paths:
            print(f"No images found in {subfolder_path}")
            continue

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            predictions = test_images_for_cats(batch_paths, model, device)
            num_images += len(predictions)
            num_cats += sum(pred > 0.5 for pred in predictions)
        
        print(f"Subfolder {subfolder_path}: {num_images} images tested, {num_cats} identified as cats.")

print("above main")
if __name__ == "__main__":
    # Load the cat detection model
    print("Loading cat detection model...")
    cat_model = load_cat_model()

    # Test images in specified subfolders
    test_images_in_subfolders(cat_model)

    # Wait for user input before ending
    input("Press Enter to exit...")