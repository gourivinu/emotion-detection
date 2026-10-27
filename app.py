from pathlib import Path
import streamlit as st
import joblib

st.set_page_config(page_title="Emotion Detection", page_icon="💗")

st.title("💗 Emotion Detection App")
st.write("Enter a sentence to predict its emotion.")

folder = Path(__file__).resolve().parent
model_path = folder / "emotion_model.pkl"
vectorizer_path = folder / "emotion_vectorizer.pkl"

text = st.text_area("Enter your sentence:", placeholder="I am so happy today!")
clicked = st.button("Detect Emotion", type="primary")

@st.cache_resource
def load_model_files(model_file, vectorizer_file):
    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)
    return model, vectorizer

model = None
vectorizer = None

if model_path.is_file() and vectorizer_path.is_file():
    try:
        model, vectorizer = load_model_files(
        str(model_path),
        str(vectorizer_path)
        )
    except Exception as error:
        st.error(f"Could not load model files: {error}")
else:
        st.warning(
        "The model files are missing. Check that emotion_model.pkl "
            "and emotion_vectorizer.pkl are uploaded to GitHub.")

if clicked:
        if not text.strip():
                st.warning("Please enter a sentence.")

        elif model is None or vectorizer is None:
                st.error("Prediction is unavailable until both model files are uploaded.")
        else:
            emotions = {
        0: "Sadness",
        1: "Joy",
        2: "Love",
        3: "Anger",
        4: "Fear",
        5: "Surprise"}
text_vector = vectorizer.transform([text])
prediction = int(model.predict(text_vector)[0])
st.success(
        f"Predicted Emotion: {emotions.get(prediction, str(prediction))}"
    )
