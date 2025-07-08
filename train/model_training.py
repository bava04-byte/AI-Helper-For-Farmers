import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# === Config ===
BASE_DIR = os.getcwd()
DATA_DIR = r"C:\crop_diagnosis_app\data\PlantVillage"  # <-- Update this if needed

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "crop_disease_cnn.pt")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
BATCH_SIZE = 32
EPOCHS = 5
IMAGE_SIZE = 224

# === Create model folder if missing ===
os.makedirs(MODEL_DIR, exist_ok=True)

# === Transforms (with normalization for ResNet) ===
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Dataset ===
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes

# Save labels for inference later
with open(LABELS_PATH, "w") as f:
    f.write("\n".join(class_names))

# === Split into train and val ===
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

# === Training Loop ===
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

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total

    print(f"✅ Epoch {epoch + 1}/{EPOCHS} - Loss: {running_loss:.4f} - Val Acc: {val_acc:.4f}")

# === Save model ===
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n✅ Training complete. Model saved to: {MODEL_PATH}")
print(f"✅ Labels saved to: {LABELS_PATH}")
