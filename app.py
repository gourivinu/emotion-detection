import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="Emotion Detection",
    page_icon="💗"
)

MODEL_PATH = "fine_tuned_emotion_model"

EMOTION_LABELS = {
    0: "SAD 😢",
    1: "HAPPY 😊",
    2: "LOVE 💗",
    3: "ANGRY 😡",
    4: "FEAR 😨",
    5: "SURPRISE 😲",
}


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model


st.title("💗 Emotion Detection")
st.write("Enter a sentence to predict its emotion.")

try:
    tokenizer, model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

text = st.text_area("Enter your sentence:")

if st.button("Detect Emotion"):
    if text.strip():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()

        emotion = EMOTION_LABELS.get(prediction, "UNKNOWN")

        st.success(f"Predicted Emotion: {emotion}")
        st.write(f"Confidence: {confidence:.2%}")
    else:
        st.warning("Please enter a sentence.")
