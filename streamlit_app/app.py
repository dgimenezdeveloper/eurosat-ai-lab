import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import joblib

st.set_page_config(page_title="EuroSAT AI Lab - Explorador", layout="wide", page_icon="🛰️")
st.title("🛰️ EuroSAT AI Lab — Consola Científica")
st.markdown("**Asignatura:** Inteligencia Artificial (UNaB 2026) | **Docente:** Lic. Pablo Moreira")

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
SPLITS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "splits.json")
BASELINE_PATH = os.path.join(ARTIFACTS_DIR, "models", "baseline_logreg.joblib")

@st.cache_resource
def load_baseline_model():
    if os.path.exists(BASELINE_PATH):
        return joblib.load(BASELINE_PATH)
    return None

tab1, tab2 = st.tabs(["📊 Métricas y Partición", "🔍 Inferencia en Vivo"])

with tab1:
    st.subheader("Partición Estratificada de Datos (80 / 10 / 10)")
    if os.path.exists(SPLITS_PATH):
        with open(SPLITS_PATH, "r") as f:
            splits = json.load(f)
        c1, c2, c3 = st.columns(3)
        c1.metric("Entrenamiento (Train 80%)", f"{len(splits['train_indices']):,} imgs")
        c2.metric("Validación (Dev 10%)", f"{len(splits['dev_indices']):,} imgs")
        c3.metric("Evaluación Final (Test 10%)", f"{len(splits['test_indices']):,} imgs")
        st.write("**Clases registradas (10):**", ", ".join(splits["classes"]))
    else:
        st.warning("No se encontró splits.json. Ejecuta setup_project.py primero.")

    summary_file = os.path.join(ARTIFACTS_DIR, "metrics", "models_summary.json")
    if os.path.exists(summary_file):
        st.subheader("Historial de Modelos")
        with open(summary_file, "r") as f:
            st.dataframe(pd.DataFrame(json.load(f)), use_container_width=True)

with tab2:
    st.subheader("Inferencia Directa con Modelo Baseline")
    model_artifact = load_baseline_model()
    
    if model_artifact is None:
        st.error("No se encontró el modelo entrenado en artifacts/models/baseline_logreg.joblib.")
    else:
        uploaded_file = st.file_uploader("Sube un parche satelital (64x64)", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            col1, col2 = st.columns([1, 2])
            img = Image.open(uploaded_file).convert("RGB")
            with col1:
                st.image(img, caption="Imagen Subida", width=180)
            
            with col2:
                # Preprocesamiento idéntico al entrenamiento
                img_resized = img.resize((64, 64))
                flat = np.array(img_resized, dtype=np.float32).flatten().reshape(1, -1)
                scaled = model_artifact["scaler"].transform(flat)
                
                probs = model_artifact["model"].predict_proba(scaled)[0]
                classes = model_artifact["classes"]
                pred_idx = int(np.argmax(probs))
                
                st.success(f"**Predicción:** {classes[pred_idx]} ({probs[pred_idx]*100:.2f}%)")
                
                df_probs = pd.DataFrame({
                    "Clase": classes,
                    "Probabilidad (%)": [round(float(p) * 100, 2) for p in probs]
                }).sort_values("Probabilidad (%)", ascending=False)
                
                st.dataframe(df_probs, use_container_width=True)