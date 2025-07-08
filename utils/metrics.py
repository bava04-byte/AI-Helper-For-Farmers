import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

def get_pytorch_model_accuracy(
    model_path="models/crop_disease_cnn.pt",
    data_dir="data/PlantVillage",  # Must match your val set
    batch_size=32
):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Validation data not found at: {data_dir}")

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load dataset
    dataset = ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    class_names = dataset.classes
    num_classes = len(class_names)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        raise RuntimeError(
            f"⚠️ Model mismatch:\n"
            f"- Expected classes: {num_classes} (based on folders in {data_dir})\n"
            f"- Error: {str(e)}"
        )

    model.to(device)
    model.eval()

    # Evaluate
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0
    return round(accuracy * 100, 2)
