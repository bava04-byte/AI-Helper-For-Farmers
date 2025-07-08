import torch
from torchvision import models
import os

# === Paths ===
original_path = r"C:\crop_diagnosis_app\train\models\crop_disease_cnn.pt"
compressed_path = r"C:\crop_diagnosis_app\train\models\crop_disease_cnn_compressed.pt"

# === Load model ===
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 13)  # Match saved model class count

# === Load weights ===
model.load_state_dict(torch.load(original_path, map_location="cpu"))

# === Save compressed model
torch.save(model.state_dict(), compressed_path, _use_new_zipfile_serialization=True)
print(f"✅ Compressed model saved to: {compressed_path}")
