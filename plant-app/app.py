from pathlib import Path
 
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
 
# ---------- Settings ----------
BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "plant_model.keras"
NAMES_PATH = BASE / "model" / "class_names.txt"
CSV_PATH = BASE / "data" / "plants.csv"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.70   # below this we say "not sure"
 
st.set_page_config(page_title="Plant Species Identifier", page_icon="🌿")
 
 
# ---------- Loading (done once, then remembered) ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)
 
 
@st.cache_data
def load_class_names():
    lines = NAMES_PATH.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]
 
 
@st.cache_data
def load_knowledge():
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    return df.set_index("class_name")
 
 
# ---------- Prediction ----------
def predict(image, model, class_names):
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img, dtype="float32"), axis=0)
    probs = model.predict(arr, verbose=0)[0]
    top3 = np.argsort(probs)[::-1][:3]
    return [(class_names[i], float(probs[i])) for i in top3]
 
 
# ---------- Page ----------
st.title("🌿 Plant Species Identifier")
st.write(
    "Upload a clear photo of a single leaf. The model predicts the species, "
    "then shows its genus and key characteristics."
)
 
model = load_model()
class_names = load_class_names()
knowledge = load_knowledge()
 
file = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])
 
if file is not None:
    image = Image.open(file)
    st.image(image, caption="Your photo", use_container_width=True)
 
    with st.spinner("Analysing..."):
        results = predict(image, model, class_names)
 
    best_name, best_conf = results[0]
 
    if best_conf < CONFIDENCE_THRESHOLD:
        st.warning(
            f"I'm not sure about this one (best guess: {best_name}, "
            f"{best_conf:.0%}). Try a clearer photo of a single leaf."
        )
    else:
        info = knowledge.loc[best_name]
        st.success(f"Prediction: {best_name} ({best_conf:.1%} confidence)")
        st.markdown(f"**Species:** *{info['species_name']}*")
        st.markdown(f"**Genus:** {info['genus']}")
        st.markdown(f"**Family:** {info['family']}")
        st.markdown(f"**Characteristics:** {info['characteristics']}")
        st.markdown(f"**Uses:** {info['uses']}")
        st.markdown(f"**Habitat:** {info['habitat']}")
 
    st.subheader("Top 3 predictions")
    for name, conf in results:
        st.write(f"{name}: {conf:.1%}")
        st.progress(conf)
 
st.caption(
    "Prototype trained on 14 crop species. It can only recognise these species "
    "and works best on clean photos of a single leaf."
)