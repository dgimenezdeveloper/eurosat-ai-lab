import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go
from torchvision.datasets import ImageFolder

# Configuración de página
st.set_page_config(
    page_title="EuroSAT AI Lab — Consola Científica",
    layout="wide",
    page_icon="🛰️",
    initial_sidebar_state="expanded"
)

# Rutas
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "eurosat", "2750")
SPLITS_PATH = os.path.join(BASE_DIR, "data", "processed", "splits.json")
BASELINE_PATH = os.path.join(BASE_DIR, "artifacts", "models", "baseline_logreg.joblib")

CLASS_NAMES_ES = {
    "AnnualCrop": "Cultivo Anual",
    "Forest": "Bosque",
    "HerbaceousVegetation": "Vegetación Herbácea",
    "Highway": "Autopista",
    "Industrial": "Zona Industrial",
    "Pasture": "Pastizal",
    "PermanentCrop": "Cultivo Permanente",
    "Residential": "Zona Residencial",
    "River": "Río",
    "SeaLake": "Mar o Lago"
}

@st.cache_resource
def load_resources():
    model_artifact = joblib.load(BASELINE_PATH) if os.path.exists(BASELINE_PATH) else None
    dataset = ImageFolder(root=DATA_RAW_DIR) if os.path.exists(DATA_RAW_DIR) else None
    splits = None
    if os.path.exists(SPLITS_PATH):
        with open(SPLITS_PATH, "r") as f:
            splits = json.load(f)
    return model_artifact, dataset, splits

model_artifact, dataset, splits = load_resources()

# Header
st.title("🛰️ EuroSAT AI Lab — Consola Científica & MLOps")
st.markdown("**Asignatura:** Inteligencia Artificial (UNaB 2026) | **Docente:** Lic. Pablo Moreira | **Grupo 5**")
st.divider()

# Pestañas Principales
tab_kpi, tab_3d, tab_cm, tab_inference = st.tabs([
    "📈 Diagnóstico Clínico & KPIs",
    "🧊 Espacio Espectral 3D (EDA)",
    "🎯 Matriz de Confusión",
    "🔍 Inferencia en Vivo & Canales"
])

# ==========================================
# PESTAÑA 1: DIAGNÓSTICO CLÍNICO & KPIS
# ==========================================
with tab_kpi:
    st.subheader("Evaluación Global del Baseline Calibrado (Dev Set)")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Métrica de Optimización (Macro F1)", "0.3524", delta="Referencia Etapa 1")
    col2.metric("Exactitud en Validación (Dev Acc)", "36.70%", delta="-62.8% vs Bayes")
    col3.metric("Latencia de Inferencia", "32.10 ms", delta="< 50 ms (Aprobado)")
    col4.metric("Tamaño del Modelo", "< 1 MB", delta="< 50 MB (Aprobado)")

    st.markdown("---")
    st.subheader("Descomposición Formal del Error (Metodología Andrew Ng - Clase 3)")
    
    # Barra apilada horizontal de descomposición
    fig_error = go.Figure()
    fig_error.add_trace(go.Bar(
        y=["Descomposición del Error"], x=[5.0], name="Error Humano / Bayes (5.0%)",
        orientation='h', marker=dict(color="#10b981")
    ))
    fig_error.add_trace(go.Bar(
        y=["Descomposición del Error"], x=[53.3], name="Sesgo Evitable (53.3% - DOMINANTE)",
        orientation='h', marker=dict(color="#ef4444")
    ))
    fig_error.add_trace(go.Bar(
        y=["Descomposición del Error"], x=[5.0], name="Brecha de Varianza (5.0%)",
        orientation='h', marker=dict(color="#3b82f6")
    ))
    fig_error.update_layout(
        barmode='stack', height=140, margin=dict(l=0, r=0, t=10, b=10),
        xaxis=dict(title="Porcentaje de Error Acumulado (%)", range=[0, 100]),
        legend=dict(orientation="h", y=-0.5)
    )
    st.plotly_chart(fig_error, use_container_width=True)

    st.error("""
    **DIAGNÓSTICO FORMAL: SESGO ALTO ESTRUCTURAL (UNDERFITTING)**  
    * **Causa:** El sesgo evitable ($53.3\%$) es 10 veces mayor que la brecha de varianza ($5.0\%$). El hiperplano lineal $W \cdot x + b$ carece de capacidad para modelar bordes y texturas.  
    * **Acción para la Etapa 2:** Es mandatorio migrar a un **Perceptrón Multicapa (MLP)** con activaciones no lineales GELU/ReLU.
    """)

# ==========================================
# PESTAÑA 2: CUBO ESPECTRAL 3D
# ==========================================
with tab_3d:
    st.subheader("🧊 Proyección de Coberturas en el Cubo de Color RGB (3D Reflectance Space)")
    st.write("Cada punto representa un parche satelital proyectado según la intensidad lumínica media de sus 3 bandas espectrales visibles. Puedes **rotar, hacer zoom y girar el cubo 3D con el mouse**.")

    if dataset is not None and splits is not None:
        with st.spinner("Calculando reflectancias de muestras..."):
            # Tomar 60 muestras de cada clase para hacer un scatter 3D fluido de 600 puntos
            sample_data = []
            class_counters = {i: 0 for i in range(10)}
            
            for idx in splits["dev_indices"]:
                img, target = dataset[idx]
                if class_counters[target] < 60:
                    arr = np.array(img, dtype=np.float32)
                    sample_data.append({
                        "Rojo (Media)": arr[:, :, 0].mean(),
                        "Verde (Media)": arr[:, :, 1].mean(),
                        "Azul (Media)": arr[:, :, 2].mean(),
                        "Clase": CLASS_NAMES_ES[splits["classes"][target]]
                    })
                    class_counters[target] += 1
                if all(c >= 60 for c in class_counters.values()):
                    break

            df_3d = pd.DataFrame(sample_data)

            fig_3d = px.scatter_3d(
                df_3d, x="Rojo (Media)", y="Verde (Media)", z="Azul (Media)",
                color="Clase", opacity=0.75, size_max=6,
                title="Distribución Espectral de Muestras Sentinel-2",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title="Canal Rojo (R)",
                    yaxis_title="Canal Verde (G)",
                    zaxis_title="Canal Azul (B)"
                ),
                height=650, margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_3d, use_container_width=True)

            st.info("""
            **Lectura Científica del Gráfico 3D:**  
            * **Mar o Lago:** Agrupada abajo a la izquierda en zonas de baja reflectancia ($R, G, B < 60$), fácilmente separable.  
            * **Bosque, Pastizal y Cultivos:** Forman un racimo superpuesto a lo largo del eje Verde, demostrando por qué una frontera plana genera confusión cruzada.
            """)

# ==========================================
# PESTAÑA 3: MATRIZ DE CONFUSIÓN
# ==========================================
with tab_cm:
    st.subheader("Matriz de Confusión Oficial (Dev Set)")
    tipo_cm = st.radio("Modalidad de Visualización:", ["Recall Normalizado (Tasa de Acierto)", "Conteos Absolutos (Muestras Reales)"], horizontal=True)

    cm_path = os.path.join(BASE_DIR, "artifacts", "metrics", "confusion_matrix_cnn.json")
    if os.path.exists(cm_path) and splits is not None:
        with open(cm_path, "r") as f:
            cm_data = np.array(json.load(f))
        
        classes_es = [CLASS_NAMES_ES[c] for c in splits["classes"]]
        
        if "Normalizado" in tipo_cm:
            z_vals = cm_data.astype('float') / cm_data.sum(axis=1)[:, np.newaxis]
            z_text = [[f"{val:.2f}" for val in row] for row in z_vals]
            colorscale = "Blues"
        else:
            z_vals = cm_data
            z_text = [[str(val) for val in row] for row in z_vals]
            colorscale = "Blues"

        fig_cm = px.imshow(
            z_vals, x=classes_es, y=classes_es,
            labels=dict(x="Predicción del Modelo", y="Clase Real (Ground Truth)", color="Valor"),
            text_auto=False, color_continuous_scale=colorscale
        )
        fig_cm.update_traces(text=z_text, texttemplate="%{text}", textfont=dict(size=11))
        fig_cm.update_layout(height=650, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_cm, use_container_width=True)

# ==========================================
# PESTAÑA 4: INFERENCIA EN VIVO & CANALES
# ==========================================
with tab_inference:
    st.subheader("Inferencia Satelital en Tiempo Real")
    
    col_input, col_view = st.columns([1, 2])
    img_to_infer = None

    with col_input:
        modo = st.radio("Método de Selección de Imagen:", ["Muestra del Dev Set", "Subir Imagen Propia"], horizontal=True)
        
        if modo == "Muestra del Dev Set" and dataset is not None and splits is not None:
            clase_elegida = st.selectbox("Selecciona Cobertura a Probar:", list(CLASS_NAMES_ES.values()))
            clase_en = [k for k, v in CLASS_NAMES_ES.items() if v == clase_elegida][0]
            target_id = splits["classes"].index(clase_en)

            if st.button("🎲 Cargar Muestra Aleatoria", use_container_width=True):
                # Filtrar índices de dev que correspondan a esa clase
                class_dev_idx = [idx for idx in splits["dev_indices"] if dataset.targets[idx] == target_id]
                chosen_idx = np.random.choice(class_dev_idx)
                img, _ = dataset[chosen_idx]
                st.session_state["current_img"] = img
                st.session_state["ground_truth"] = clase_elegida

            if "current_img" in st.session_state:
                img_to_infer = st.session_state["current_img"]
                st.info(f"Ground Truth (Clase Real): **{st.session_state.get('ground_truth', '')}**")

        else:
            up_file = st.file_uploader("Sube un parche (.png, .jpg)", type=["png", "jpg", "jpeg"])
            if up_file:
                img_to_infer = Image.open(up_file).convert("RGB")

    with col_view:
        if img_to_infer is not None and model_artifact is not None:
            # Mostrar Parche agrandado y descomposición RGB
            arr_img = np.array(img_to_infer.resize((64, 64)))
            
            c_orig, c_r, c_g, c_b = st.columns(4)
            c_orig.image(img_to_infer.resize((150, 150), Image.NEAREST), caption="Parche Compuesto", width=140)
            c_r.image(arr_img[:, :, 0], caption="Canal Rojo (R)", clamp=True, width=140)
            c_g.image(arr_img[:, :, 1], caption="Canal Verde (G)", clamp=True, width=140)
            c_b.image(arr_img[:, :, 2], caption="Canal Azul (B)", clamp=True, width=140)

            # Inferencia
            flat = arr_img.astype(np.float32).flatten().reshape(1, -1)
            scaled = model_artifact["scaler"].transform(flat)
            probs = model_artifact["model"].predict_proba(scaled)[0]
            classes = model_artifact["classes"]
            pred_idx = int(np.argmax(probs))

            st.success(f"**Predicción del Modelo:** {CLASS_NAMES_ES[classes[pred_idx]]} ({classes[pred_idx]}) — **Confianza:** {probs[pred_idx]*100:.2f}%")

            # Gráfico de barras horizontales
            df_bar = pd.DataFrame({
                "Clase": [CLASS_NAMES_ES[c] for c in classes],
                "Probabilidad (%)": probs * 100
            }).sort_values("Probabilidad (%)", ascending=True)

            fig_bar = px.bar(
                df_bar, x="Probabilidad (%)", y="Clase", orientation='h',
                text=df_bar["Probabilidad (%)"].apply(lambda x: f"{x:.1f}%"),
                color="Probabilidad (%)", color_continuous_scale="Viridis",
                range_x=[0, 100]
            )
            fig_bar.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)