import os
import torch
import streamlit as st
from torchvision import transforms, models
from PIL import Image
from deep_translator import GoogleTranslator

# === Paths ===
ROOT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.abspath(os.path.join(ROOT_DIR, "..", "models", "crop_disease_cnn.pt"))
LABELS_PATH = os.path.abspath(os.path.join(ROOT_DIR, "..", "labels.txt"))

# === Load class names ===
with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Cache Model ===
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

# === Cache Transforms ===
@st.cache_data
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

# === Translate to Malayalam ===
def translate_to_malayalam(text):
    try:
        return GoogleTranslator(source='auto', target='ml').translate(text)
    except:
        return "⚠️ Malayalam translation failed."

# === Classify Image ===
def classify_image(image: Image.Image, model, transform) -> str:
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]

# === Analyze Text with Rule-based NLP ===
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

# === Suggest Solution ===
def get_solution(disease: str, symptoms: str, crop: str) -> str:
    solution = ""

    if "late blight" in disease:
        solution = (
            "Late blight is a serious fungal disease.\n\n"
            "👉 Spray **Metalaxyl + Mancozeb** or **Chlorothalonil** fungicide.\n"
            "🌱 Remove infected leaves and debris.\n"
            "💧 Avoid wetting leaves during watering."
        )
    elif "early blight" in disease:
        solution = (
            "Early blight causes spots and yellowing.\n\n"
            "👉 Use **Mancozeb** or **Azoxystrobin** fungicide.\n"
            "🌿 Prune lower leaves and avoid crowding.\n"
            "💧 Water at soil level, not leaves."
        )
    elif "bacterial" in symptoms or "black spot" in symptoms:
        solution = (
            "Suspected bacterial spot or speck.\n\n"
            "👉 Use **copper-based fungicide** weekly.\n"
            "🌿 Remove infected leaves.\n"
            "🚫 Avoid overhead watering."
        )
    elif "wilting" in symptoms:
        solution = (
            "Wilting could be from bacterial wilt or root rot.\n\n"
            "💧 Ensure proper drainage.\n"
            "🚫 Remove severely wilted plants.\n"
            "🌱 Avoid overwatering."
        )
    elif "insect" in symptoms or "hole" in symptoms:
        solution = (
            "Insect attack detected.\n\n"
            "👉 Apply **Neem oil** (organic) or **Imidacloprid** (chemical).\n"
            "🕵️‍♂️ Check leaves for whiteflies or caterpillars.\n"
            "🌅 Spray in the early morning or evening."
        )
    elif "nitrogen" in symptoms or "yellow" in symptoms:
        solution = (
            "Likely nutrient deficiency.\n\n"
            "👉 Apply **Urea** or **organic compost**.\n"
            "🌱 Try **vermicompost** or cow dung slurry."
        )
    elif "fungal" in symptoms or "mold" in symptoms:
        solution = (
            "Fungal infection suspected.\n\n"
            "👉 Use **Carbendazim** or **Mancozeb** spray.\n"
            "🌿 Remove affected parts and reduce humidity."
        )
    else:
        solution = (
            "⚠️ Unable to suggest an exact solution.\n\n"
            "📸 Please try a clearer image or provide more symptom details like color, spots, or patterns."
        )

    # Crop-specific tips
    crop = crop.lower()
    if "tomato" in crop:
        solution += "\n\n🍅 *Tomato Tip*: Rotate crops yearly and stake plants to avoid soil contact."
    elif "potato" in crop:
        solution += "\n\n🥔 *Potato Tip*: Use certified seed tubers and plant in well-drained soil."
    elif "chili" in crop:
        solution += "\n\n🌶️ *Chili Tip*: Ensure spacing and avoid waterlogging."
    elif "banana" in crop:
        solution += "\n\n🍌 *Banana Tip*: Apply potassium-rich fertilizer regularly."
    elif "brinjal" in crop:
        solution += "\n\n🍆 *Brinjal Tip*: Control stem borers early and use neem extract spray."

    return solution

# === Main Analyzer Function ===
def analyze_crop_issue(image: Image.Image, text: str, crop: str, show_malayalam: bool = False):
    model = load_model()
    transform = get_transform()

    vision_result = classify_image(image, model, transform)
    text = text.strip()
    crop = crop.lower()

    text_result = analyze_text(text, crop) if text else "No description provided. Diagnosis is based on image only."
    diagnosis = f"Image shows: **{vision_result}**\nVoice/Text suggests: **{text_result}**"

    disease = vision_result.lower()
    symptoms = text_result.lower()
    solution = get_solution(disease, symptoms, crop)

    if show_malayalam:
        mal_solution = translate_to_malayalam(solution)
        solution += "\n\n🌐 Malayalam:\n" + mal_solution

    return diagnosis, solution
