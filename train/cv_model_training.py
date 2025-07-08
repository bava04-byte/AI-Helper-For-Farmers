import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# === Config ===
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data", "PlantVillage")
MODEL_DIR = os.path.join(BASE_DIR, "models")  # Ensures model is saved inside the project
MODEL_PATH = os.path.join(MODEL_DIR, "crop_disease_cnn.pt")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
BATCH_SIZE = 32
EPOCHS = 5
IMAGE_SIZE = 224

# === Create model folder if missing ===
os.makedirs(MODEL_DIR, exist_ok=True)

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# === Dataset ===
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes

# === Save labels to file ===
with open(LABELS_PATH, "w") as f:
    f.write("\n".join(class_names))

# === Split data ===
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# === Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === Training ===
print("\U0001F4E2 Starting training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()      
        optimizer.step()
        running_loss += loss.item()

    print(f"\u2705 Epoch {epoch + 1}/{EPOCHS} - Loss: {running_loss:.4f}")

# === Save model ===
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n\u2705 Training complete. Model saved to: {MODEL_PATH}")
