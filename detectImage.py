import os
import cv2
import torch
import torchvision
from torchvision import transforms, models
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Paths
CAT_DIR = "data/natural_images/cat"
NON_CAT_DIRS = ["data/natural_images/airplane", "data/natural_images/car", "data/natural_images/dog", "data/natural_images/flower", "data/natural_images/fruit", "data/natural_images/motorbike", "data/natural_images/person"]
MODEL_OUTPUT_PATH = "models/cat_detection_model.pth"

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

class CatDataset(Dataset):
    def __init__(self, cat_dir, non_cat_dirs, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load cat images
        for root, _, files in os.walk(cat_dir):
            for file in files:
                if file.endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(root, file))
                    self.labels.append(1)  # 1 for cat

        # Load non-cat images
        for non_cat_dir in non_cat_dirs:
            for root, _, files in os.walk(non_cat_dir):
                for file in files:
                    if file.endswith((".jpg", ".jpeg", ".png")):
                        self.image_paths.append(os.path.join(root, file))
                        self.labels.append(0)  # 0 for non-cat

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformations for preprocessing images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the model
def create_cat_detection_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    return model

# Training function
def train_model(model, train_loader, val_loader, epochs=10):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
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
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_accuracy = correct / total * 100
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_accuracy = val_correct / val_total * 100
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%")
    
    return model

if __name__ == "__main__":
    print("Loading image data...")
    dataset = CatDataset(CAT_DIR, NON_CAT_DIRS, transform=transform)
    
    # Split the dataset into train and validation sets
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    print("Creating cat detection model...")
    cat_model = create_cat_detection_model()
    
    print("Training the cat detection model...")
    trained_cat_model = train_model(cat_model, train_loader, val_loader)
    
    print(f"Saving the trained model to {MODEL_OUTPUT_PATH}...")
    torch.save(trained_cat_model.state_dict(), MODEL_OUTPUT_PATH)
