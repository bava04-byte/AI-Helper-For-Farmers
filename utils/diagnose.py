import os
import torch
import streamlit as st
from torchvision import transforms, models
from PIL import Image
from deep_translator import GoogleTranslator

# === Paths ===
ROOT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.abspath(os.path.join(ROOT_DIR, "..", "train", "models", "crop_disease_cnn.pt"))
LABELS_PATH = os.path.abspath(os.path.join(ROOT_DIR, "..", "train", "labels.txt"))

# === Load class names ===
with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load compressed model ===
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

# === Preprocessing transform ===
@st.cache_data
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


# === Translate to Malayalam ===
def translate_to_malayalam(text):
    try:
        return GoogleTranslator(source='auto', target='ml').translate(text)
    except Exception:
        return "⚠️ Malayalam translation failed."

# === Image classifier ===
def classify_image(image: Image.Image, model, transform) -> str:
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]

# === Rule-based NLP analyzer ===
def analyze_text(text: str, crop: str) -> str:
    text = text.lower()
    if "yellow" in text and "leaf" in text:
        return "Possible nitrogen deficiency"
    elif "white spot" in text or "powder" in text or "mold" in text:
        return "Possible fungal infection"
    elif "dry" in text or "wilting" in text:
        return "Possible dehydration or root stress"
    elif "black spot" in text or "dark lesion" in text:
        return "Possible bacterial or fungal disease"
    elif "hole" in text or "chewed" in text or "pest" in text:
        return "Possible insect infestation"
    elif "curl" in text or "twist" in text:
        return "Possible viral infection or micronutrient deficiency"
    else:
        return "Unable to interpret description. Try giving more detail."

# === Generate action suggestions ===
def get_solution(disease: str, symptoms: str, crop: str) -> str:
    solution = ""

    if "late blight" in disease:
        solution = (
            "Late blight is a serious fungal disease.\n\n"
            "👉 Spray **Metalaxyl + Mancozeb** or **Chlorothalonil**.\n"
            "🌱 Remove infected leaves.\n"
            "💧 Avoid wetting leaves while watering."
        )
    elif "early blight" in disease:
        solution = (
            "Early blight causes leaf spots and yellowing.\n\n"
            "👉 Use **Mancozeb** or **Azoxystrobin**.\n"
            "🌿 Prune lower leaves.\n"
            "💧 Water at soil level only."
        )
    elif "bacterial" in symptoms or "black spot" in symptoms:
        solution = (
            "Suspected bacterial disease.\n\n"
            "👉 Use **copper-based fungicide** weekly.\n"
            "🌿 Remove affected leaves.\n"
            "🚫 Avoid overhead watering."
        )
    elif "wilting" in symptoms:
        solution = (
            "Wilting may be due to bacterial wilt or root rot.\n\n"
            "💧 Ensure good drainage.\n"
            "🚫 Remove severely affected plants.\n"
            "🌱 Avoid overwatering."
        )
    elif "insect" in symptoms or "hole" in symptoms:
        solution = (
            "Signs of insect damage.\n\n"
            "👉 Spray **Neem oil** or **Imidacloprid**.\n"
            "🕵️ Check leaves for pests.\n"
            "🌄 Spray during early morning or evening."
        )
    elif "nitrogen" in symptoms or "yellow" in symptoms:
        solution = (
            "Likely nutrient deficiency.\n\n"
            "👉 Use **Urea**, compost, or vermicompost.\n"
            "🌱 Consider adding cow dung slurry."
        )
    elif "fungal" in symptoms or "mold" in symptoms:
        solution = (
            "Possible fungal infection.\n\n"
            "👉 Use **Carbendazim** or **Mancozeb**.\n"
            "🌿 Prune infected areas and reduce humidity."
        )
    else:
        solution = (
            "⚠️ Unable to suggest a clear solution.\n\n"
            "📸 Try giving a clearer image or more symptom details."
        )

    # Add crop-specific tip
    crop = crop.lower()
    if "tomato" in crop:
        solution += "\n\n🍅 *Tomato Tip*: Rotate crops and stake plants to avoid soil contact."
    elif "potato" in crop:
        solution += "\n\n🥔 *Potato Tip*: Use certified tubers and avoid waterlogging."
    elif "chili" in crop:
        solution += "\n\n🌶️ *Chili Tip*: Space plants well and check for thrips or mites."
    elif "banana" in crop:
        solution += "\n\n🍌 *Banana Tip*: Apply potassium-rich fertilizer regularly."
    elif "brinjal" in crop:
        solution += "\n\n🍆 *Brinjal Tip*: Control stem borers early with neem spray."

    return solution

# === Main Analyzer Function ===
def analyze_crop_issue(image: Image.Image, text: str, crop: str, show_malayalam: bool = False):
    model = load_model()
    transform = get_transform()

    vision_result = classify_image(image, model, transform)
    crop = crop.lower()
    text = text.strip()

    text_result = analyze_text(text, crop) if text else "No description provided."
    diagnosis = f"Image shows: **{vision_result}**\nVoice/Text suggests: **{text_result}**"

    disease = vision_result.lower()
    symptoms = text_result.lower()
    solution = get_solution(disease, symptoms, crop)

    if show_malayalam:
        mal_solution = translate_to_malayalam(solution)
        solution += "\n\n🌐 Malayalam:\n" + mal_solution

    return diagnosis, solution
