import os
import cv2
import torch
import torchvision
from torchvision import transforms, models
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# Paths
IMAGE_DATA_DIR = "data/image_data"
MODEL_OUTPUT_PATH = "models/cat_detection_model.pth"

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Custom Dataset to Handle .cat Files and Crop Images
class CatDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.annotation_paths = []

        # Collect image and corresponding .cat file paths
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".jpg"):
                    image_path = os.path.join(root, file)
                    annotation_path = f"{image_path}.cat"
                    if os.path.exists(annotation_path):
                        self.image_paths.append(image_path)
                        self.annotation_paths.append(annotation_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and its corresponding .cat file
        image_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]
        image = cv2.imread(image_path)
        with open(annotation_path, "r") as f:
            annotation = list(map(int, f.read().strip().split()))

        # Extract bounding box from annotation (left eye to right ear-3)
        points = annotation[1:]  # Skip the first value (number of points)
        x_min = min(points[0::2])  # Min of all x-coordinates
        y_min = min(points[1::2])  # Min of all y-coordinates
        x_max = max(points[0::2])  # Max of all x-coordinates
        y_max = max(points[1::2])  # Max of all y-coordinates

        # Crop image around bounding box
        cropped_image = image[y_min:y_max, x_min:x_max]

        # Apply transformations if provided
        if self.transform:
            cropped_image = self.transform(cropped_image)

        # Label is always 1 (since these are all cat images)
        label = 1

        return cropped_image, label

# Define transformations for preprocessing images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the model
def create_cat_detection_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)  # Binary classification (cat or not cat)
    return model

# Training function
def train_model(model, train_loader, epochs=5):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = torch.sigmoid(outputs) > 0.5
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total * 100
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

    return model

if __name__ == "__main__":
    print("Loading image data...")
    dataset = CatDataset(IMAGE_DATA_DIR, transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("Creating cat detection model...")
    cat_model = create_cat_detection_model()

    print("Training the cat detection model...")
    trained_cat_model = train_model(cat_model, train_loader)

    print(f"Saving the trained model to {MODEL_OUTPUT_PATH}...")
    torch.save(trained_cat_model.state_dict(), MODEL_OUTPUT_PATH)
