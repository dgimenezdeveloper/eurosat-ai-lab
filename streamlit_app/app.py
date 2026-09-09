import streamlit as st
import json
import os
from PIL import Image
import requests

st.set_page_config(page_title="EuroSAT AI Lab - Explorador", layout="wide")

st.title("🛰️ EuroSAT AI Lab - Consola de Exploración Científica")
st.markdown("**Asignatura:** Inteligencia Artificial (UNaB) | **Docente:** Lic. Pablo Moreira")

tab1, tab2 = st.tabs(["📊 Métricas y Splits", "🔍 Inferencia Rápida"])

with tab1:
    st.subheader("Partición de Datos Estratificada (80 / 10 / 10)")
    if os.path.exists("./data/processed/splits.json"):
        with open("./data/processed/splits.json", "r") as f:
            splits = json.load(f)
        c1, c2, c3 = st.columns(3)
        c1.metric("Train (80%)", len(splits["train_indices"]))
        c2.metric("Dev (10%)", len(splits["dev_indices"]))
        c3.metric("Test (10%)", len(splits["test_indices"]))
        st.write("**Clases registradas:**", ", ".join(splits["classes"]))
    else:
        st.warning("No se encontró el archivo splits.json")

with tab2:
    st.subheader("Prueba de Inferencia con Modelo")
    uploaded_file = st.file_uploader("Sube una imagen satelital (.jpg, .png)", type=["jpg", "png", "tif"])
    if uploaded_file:
        st.image(uploaded_file, caption="Imagen cargada", width=200)
        if st.button("Clasificar"):
            try:
                res = requests.post("http://localhost:8000/predict", files={"file": uploaded_file.getvalue()})
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"Predicción: **{data['predicted_class']}** ({data['confidence']}%)")
                    st.json(data["ranked_probabilities"])
                else:
                    st.error("Error en la API")
            except Exception as e:
                st.warning(f"La API de FastAPI aún no está corriendo en el puerto 8000 ({e})")
