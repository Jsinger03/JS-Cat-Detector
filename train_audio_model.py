# Filename: train_audio_model.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Use MPS if available; otherwise, fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load processed data
DATA_PATH = os.path.join("audio_data", "processed", "audio_dataset.npz")
if not os.path.exists(DATA_PATH):
    print(f"Processed data not found at {DATA_PATH}. Run preprocess_audio.py first.")
    exit(1)
data = np.load(DATA_PATH, allow_pickle=True)
X_train = data["X_train"]  # shape: (N_train, n_mels, time_frames)
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

# Expand dimensions: add channel dimension (N, 1, n_mels, time_frames)
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

# After data loading, before creating tensors
def add_noise(x, noise_level=0.1):
    noise = np.random.normal(0, noise_level, x.shape)
    x_noisy = x + noise
    return np.clip(x_noisy, 0, 1)

# Add noise to training data only
X_train = add_noise(X_train)

# Convert to torch tensors and move to device
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# DataLoaders
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define a lightweight CNN for audio classification
class MeowCNN(nn.Module):
    def __init__(self, t_dim):
        super(MeowCNN, self).__init__()
        self.dropout = nn.Dropout(0.5)  # Add significant dropout
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # After 3 poolings, height becomes 128/8 = 16; width becomes t_dim/8.
        self.fc1 = nn.Linear(64 * 16 * (t_dim // 8), 64)
        self.fc2 = nn.Linear(64, 2)  # Two classes: 0 (non-meow), 1 (meow)

    def forward(self, x):
        x = self.dropout(self.pool(torch.relu(self.bn1(self.conv1(x)))))
        x = self.dropout(self.pool(torch.relu(self.bn2(self.conv2(x)))))
        x = self.dropout(self.pool(torch.relu(self.bn3(self.conv3(x)))))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Determine t_dim from training data shape
t_dim = X_train_tensor.shape[3]
model = MeowCNN(t_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# After loading data
print("\nData Validation:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train distribution: {np.bincount(y_train)}")
print(f"y_test distribution: {np.bincount(y_test)}")
print(f"X_train range: [{X_train.min():.3f}, {X_train.max():.3f}]")
print(f"X_train mean: {X_train.mean():.3f}, std: {X_train.std():.3f}")

# After loading data, add this to examine the actual features
print("\nFeature Analysis:")
print(f"Number of zero features: {np.sum(X_train == 0)}/{X_train.size}")
print(f"Number of features close to 1: {np.sum(X_train > 0.99)}/{X_train.size}")

# After creating tensors
print("\nDevice Validation:")
print(f"X_train_tensor device: {X_train_tensor.device}")
print(f"Model device: {next(model.parameters()).device}")

# After model definition, add this helper function
def get_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    # In training loop, first batch only
    first_batch = True
    for inputs, labels in train_loader:
        if first_batch:
            print("\nFirst Batch Details:")
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            print(f"Predictions (first 5):")
            for i in range(5):
                print(f"True: {labels[i].item()}, Probs: {probs[i].tolist()}")
            
            loss = criterion(outputs, labels)
            loss.backward()
            grad_norm = get_gradient_norm(model)
            print(f"Gradient norm: {grad_norm}")
            optimizer.zero_grad()
            
            first_batch = False
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
    test_loss /= test_total
    test_acc = test_correct / test_total * 100
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")

# Save the trained model weights
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "meow_classifier.pth")
torch.save(model.state_dict(), model_path)
print(f"Audio model saved to {model_path}")