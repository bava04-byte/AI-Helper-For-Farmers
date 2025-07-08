import streamlit as st
from PIL import Image
from utils.diagnose import analyze_crop_issue
from utils.metrics import get_pytorch_model_accuracy
import speech_recognition as sr
import tempfile
import os
from googletrans import Translator

# --- Page Config ---
st.set_page_config(page_title="Smart Crop Issue Detector", layout="centered")
st.title("🌾 Smart Crop Issue Detector")
st.markdown("Upload an image of your crop and describe the problem using text or voice.")

# --- Sidebar Accuracy ---
with st.sidebar:
    st.markdown("## 📊 Model Accuracy")
    try:
        acc = get_pytorch_model_accuracy()
        st.success(f"🖼️ Image Model Accuracy: {acc}%")
    except Exception as e:
        st.error(f"⚠️ Accuracy check failed: {e}")

# --- Crop Type Selection ---
crop_type = st.selectbox("🌱 Select Crop Type", ["Tomato", "Potato"])

# --- Language Selection ---
language_map = {
    "English": "en",
    "Malayalam": "ml",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn"
}
selected_lang = st.selectbox("🌐 Choose Output Language", list(language_map.keys()))
lang_code = language_map[selected_lang]

# --- Image Upload Section ---
with st.expander("🖼️ Upload Crop Image"):
    uploaded_file = st.file_uploader("Upload Crop Image", type=["jpg", "jpeg", "png"])

# --- Text Description ---
st.subheader("✍️ Describe the Problem (Recommended)")
description = st.text_area(
    "Type crop symptoms (e.g., yellow leaves, black spots, white powder, curling, dryness, insects)..."
)

# --- Voice Input Section ---
st.subheader("🎙️ Or Upload Voice Description (Optional)")
voice_input_text = ""

with st.expander("📁 Upload a Voice Note (WAV Only)"):
    voice_file = st.file_uploader("Upload Voice Note (WAV)", type=["wav"], key="voice_input")
    if voice_file:
        recognizer = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(voice_file.read())
            tmp_path = tmp.name
        try:
            with sr.AudioFile(tmp_path) as source:
                audio = recognizer.record(source)
                voice_input_text = recognizer.recognize_google(audio, language="en-IN")
                st.success(f"🗣️ Voice to Text: {voice_input_text}")
        except Exception as e:
            st.error(f"Voice recognition failed: {e}")

# --- Translator Function ---
def translate_text(text, lang_code):
    try:
        translator = Translator()
        return translator.translate(text, dest=lang_code).text
    except Exception as e:
        return f"⚠️ Translation failed: {e}"

# --- Diagnosis Section ---
st.header("🔬 Get Diagnosis")
if st.button("Analyze Now"):
    if not uploaded_file:
        st.warning("Please upload a crop image.")
    else:
        image = Image.open(uploaded_file)
        final_input = description.strip() if description.strip() else voice_input_text.strip()

        if not final_input:
            st.warning("Please describe the issue using text or voice.")
        else:
            with st.spinner("Analyzing the crop issue..."):
                diagnosis, solution = analyze_crop_issue(image, final_input, crop_type)

            if lang_code != "en":
                diagnosis = translate_text(diagnosis, lang_code)
                solution = translate_text(solution, lang_code)

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="📷 Uploaded Crop Image", use_container_width=True)

            with col2:
                st.markdown("### 🧪 Diagnosis")
                st.success(diagnosis)

                st.markdown("### 💡 Suggested Action")
                st.info(solution)
