import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn.functional as F

# ==========================================
# 0. CONFIGURACIÓN DE PÁGINA Y RUTAS
# ==========================================
st.set_page_config(
    page_title="EuroSAT AI Lab — Consola Científica & MLOps",
    layout="wide",
    page_icon="🛰️",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import (
    DATA_RAW_DIR, SPLITS_PATH, ARTIFACTS_DIR,
    CLASS_NAMES, CLASS_NAMES_ES, DEVICE
)
from src.models.mlp import EurosatMLP

BASELINE_PATH = os.path.join(ARTIFACTS_DIR, "models", "baseline_logreg.joblib")
MLP_PATH = os.path.join(ARTIFACTS_DIR, "models", "mlp_regularized.pt")
CM_PATH = os.path.join(ARTIFACTS_DIR, "metrics", "confusion_matrix_cnn.json")
LEDGER_PATH = os.path.join(ARTIFACTS_DIR, "metrics", "experiments_ledger.json")
ERRORS_METADATA_PATH = os.path.join(ARTIFACTS_DIR, "error_samples", "errors_50_metadata.json")
ERRORS_IMG_DIR = os.path.join(ARTIFACTS_DIR, "error_samples")

# ==========================================
# 1. CARGA CACHEADA DE RECURSOS
# ==========================================
@st.cache_resource
def load_resources():
    baseline_artifact = joblib.load(BASELINE_PATH) if os.path.exists(BASELINE_PATH) else None
    
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder(root=DATA_RAW_DIR) if os.path.exists(DATA_RAW_DIR) else None
    
    splits = None
    if os.path.exists(SPLITS_PATH):
        with open(SPLITS_PATH, "r", encoding="utf-8") as f:
            splits = json.load(f)

    mlp_model = None
    if os.path.exists(MLP_PATH):
        try:
            mlp_model = EurosatMLP(input_dim=12288, num_classes=10, regularized=True, dropout_rate=0.35)
            mlp_model.load_state_dict(torch.load(MLP_PATH, map_location=DEVICE))
            mlp_model.to(DEVICE)
            mlp_model.eval()
        except Exception:
            mlp_model = None

    ledger_data = []
    if os.path.exists(LEDGER_PATH):
        try:
            with open(LEDGER_PATH, "r", encoding="utf-8") as f:
                ledger_data = json.load(f)
        except Exception:
            ledger_data = []

    errors_metadata = []
    if os.path.exists(ERRORS_METADATA_PATH):
        try:
            with open(ERRORS_METADATA_PATH, "r", encoding="utf-8") as f:
                errors_metadata = json.load(f)
        except Exception:
            errors_metadata = []

    return baseline_artifact, dataset, splits, mlp_model, ledger_data, errors_metadata

baseline_artifact, dataset, splits, mlp_model, ledger_data, errors_metadata = load_resources()

# ==========================================
# 2. SIDEBAR: NAVEGACIÓN EXCLUSIVA POR ETAPAS
# ==========================================
st.sidebar.title("🛰️ EuroSAT AI Lab")
st.sidebar.markdown("**Consola Científica & MLOps**")
st.sidebar.caption("Inteligencia Artificial (UNaB 2026) | **Lic. Pablo Moreira** | **Grupo 5**")
st.sidebar.divider()

# Selector de Etapas en el Sidebar
etapa = st.sidebar.radio(
    "Navegación del Trabajo Práctico:",
    [
        "🛰️ Etapa 1: Baseline Lineal & EDA",
        "🧠 Etapa 2: Red Densa (MLP) & Auditoría",
        "🔬 Etapa 3: Roadmap Convolucional (CNN)",
        "📊 MLOps Ledger & Experimentos"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("Telemetría de Artefactos")
st.sidebar.markdown(f"• **EuroSAT Raw:** {'✅ Conectado' if dataset else '❌ Ausente'}")
st.sidebar.markdown(f"• **Particiones (80/10/10):** {'✅ Generadas' if splits else '❌ No halladas'}")
st.sidebar.markdown(f"• **Baseline (LogReg):** {'✅ Serializado' if baseline_artifact else '⚠️ Pendiente'}")
st.sidebar.markdown(f"• **MLP PyTorch:** {'✅ Checkpoint cargado' if mlp_model else '⚠️ Sin entrenar/guardar'}")
st.sidebar.markdown(f"• **50 Errores Auditados:** {f'✅ {len(errors_metadata)} muestras' if errors_metadata else '⚠️ 0 muestras'}")

# Header Global
st.title("🛰️ EuroSAT AI Lab — Consola Científica & MLOps")
st.markdown("**Asignatura:** Inteligencia Artificial (UNaB 2026) | **Docente:** Lic. Pablo Moreira | **Grupo 5**")
st.divider()

# ==============================================================================
# ETAPA 1: PESTAÑAS (TABS) ORIGINALES RESTAURADAS AL 100%
# ==============================================================================
if etapa == "🛰️ Etapa 1: Baseline Lineal & EDA":
    tab_kpi, tab_3d, tab_cm, tab_inference = st.tabs([
        "📈 Diagnóstico Clínico & KPIs",
        "🧊 Espacio Espectral 3D (EDA)",
        "🎯 Matriz de Confusión",
        "🔍 Inferencia en Vivo & Canales"
    ])

    # ----------------------------------------------------
    # PESTAÑA 1: DIAGNÓSTICO CLÍNICO & KPIS
    # ----------------------------------------------------
    with tab_kpi:
        st.subheader("Evaluación Global del Baseline Calibrado (Dev Set)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Métrica de Optimización (Macro F1)", "0.3524", delta="Referencia Etapa 1")
        col2.metric("Exactitud en Validación (Dev Acc)", "36.70%", delta="-62.8% vs Bayes")
        col3.metric("Latencia de Inferencia", "32.10 ms", delta="< 50 ms (Aprobado)")
        col4.metric("Tamaño del Modelo", "< 1 MB", delta="< 50 MB (Aprobado)")
        st.markdown("---")
        st.subheader("Descomposición Formal del Error (Metodología Andrew Ng - Clase 3)")
        
        # Barra apilada horizontal original
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

    # ----------------------------------------------------
    # PESTAÑA 2: CUBO ESPECTRAL 3D (EDA)
    # ----------------------------------------------------
    with tab_3d:
        st.subheader("🧊 Proyección de Coberturas en el Cubo de Color RGB (3D Reflectance Space)")
        st.write("Cada punto representa un parche satelital proyectado según la intensidad lumínica media de sus 3 bandas espectrales visibles. Puedes **rotar, hacer zoom y girar el cubo 3D con el mouse**.")
        if dataset is not None and splits is not None:
            with st.spinner("Calculando reflectancias de muestras..."):
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

    # ----------------------------------------------------
    # PESTAÑA 3: MATRIZ DE CONFUSIÓN OFICIAL
    # ----------------------------------------------------
    with tab_cm:
        st.subheader("Matriz de Confusión Oficial (Dev Set)")
        tipo_cm = st.radio("Modalidad de Visualización:", ["Recall Normalizado (Tasa de Acierto)", "Conteos Absolutos (Muestras Reales)"], horizontal=True)
        if os.path.exists(CM_PATH) and splits is not None:
            with open(CM_PATH, "r") as f:
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

    # ----------------------------------------------------
    # PESTAÑA 4: INFERENCIA EN VIVO & CANALES
    # ----------------------------------------------------
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
                    st.session_state["ground_truth"] = "Subida por Usuario"
        with col_view:
            if img_to_infer is not None and baseline_artifact is not None:
                arr_img = np.array(img_to_infer.resize((64, 64)))
                c_orig, c_r, c_g, c_b = st.columns(4)
                c_orig.image(img_to_infer.resize((150, 150), Image.NEAREST), caption="Parche Compuesto", width=140)
                c_r.image(arr_img[:, :, 0], caption="Canal Rojo (R)", clamp=True, width=140)
                c_g.image(arr_img[:, :, 1], caption="Canal Verde (G)", clamp=True, width=140)
                c_b.image(arr_img[:, :, 2], caption="Canal Azul (B)", clamp=True, width=140)
                flat = arr_img.astype(np.float32).flatten().reshape(1, -1)
                scaled = baseline_artifact["scaler"].transform(flat)
                probs = baseline_artifact["model"].predict_proba(scaled)[0]
                classes = baseline_artifact["classes"]
                pred_idx = int(np.argmax(probs))
                st.success(f"**Predicción del Modelo:** {CLASS_NAMES_ES[classes[pred_idx]]} ({classes[pred_idx]}) — **Confianza:** {probs[pred_idx]*100:.2f}%")
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

# ==============================================================================
# ETAPA 2: RED DENSA (MLP) & AUDITORÍA DE ERRORES (ORGANIZADA EN TABS)
# ==============================================================================
elif etapa == "🧠 Etapa 2: Red Densa (MLP) & Auditoría":
    tab_audit, tab_arch, tab_mlp_infer = st.tabs([
        "🔬 Auditoría de 50 Errores del Baseline",
        "🏗️ Arquitectura MLP & Curvas",
        "🔍 Inferencia MLP en Vivo"
    ])

    with tab_audit:
        st.subheader("🔬 Auditoría Clínica de 50 Muestras de Error (Sección 4.2.1)")
        st.write("Análisis sistemático de fallas en el Dev Set para fundamentar las mejoras arquitectónicas hacia el MLP y CNN.")
        if errors_metadata:
            df_err = pd.DataFrame(errors_metadata)
            col_f1, col_f2 = st.columns([2, 1])
            with col_f1:
                categorias = ["Todas"] + list(df_err["error_category"].unique())
                cat_sel = st.selectbox("Filtrar por Causa Diagnóstica:", categorias)
            with col_f2:
                st.metric("Total de Muestras Auditadas", f"{len(df_err)} casos", delta="Dev Set")
            filtered_err = df_err if cat_sel == "Todas" else df_err[df_err["error_category"] == cat_sel]
            st.caption(f"Mostrando **{len(filtered_err)}** casos de error correspondientes:")
            
            # Grilla en 3 columnas con tarjetas delimitadas y escala nativa
            cols = st.columns(3)
            for idx, (_, r) in enumerate(filtered_err.iterrows()):
                col = cols[idx % 3]
                img_p = os.path.join(ERRORS_IMG_DIR, r["image_file"])
                with col:
                    with st.container(border=True):
                        c_header1, c_header2 = st.columns([3, 2])
                        c_header1.markdown(f"**Caso #{r['error_id']:02d}**")
                        c_header2.caption("🛰️ 64×64 px")
                        
                        if os.path.exists(img_p):
                            # Escala controlada (110px) sin estiramiento deformante
                            st.image(Image.open(img_p), width=110)
                        
                        st.markdown(f"**Real:** `{r['real_class_es']}`")
                        st.markdown(f"**Pred:** :red[**{r['predicted_class_es']}**] ({r['confidence_predicted']}%)")
                        st.caption(f"🏷️ *{r['error_category']}*")
                        st.info(f"💡 {r['clinical_justification']}")
        else:
            st.warning("No se encontró `errors_50_metadata.json`. Ejecutá `python scripts/extract_baseline_errors.py` en tu terminal para generar la auditoría.")

    with tab_arch:
        st.subheader("Arquitectura del Perceptrón Multicapa (MLP) y Diagnóstico Clínico")
        st.code("""
Entrada: 12.288 dimensiones (64x64x3 píxeles aplanados)
 ├── Capa 1: Linear(12288 -> 512) + BatchNorm1d + GELU + Dropout(0.35)
 ├── Capa 2: Linear(512 -> 256)   + BatchNorm1d + GELU + Dropout(0.35)
 ├── Capa 3: Linear(256 -> 128)   + BatchNorm1d + GELU + Dropout(0.35)
 └── Salida: Linear(128 -> 10)     (Logits para CrossEntropyLoss)
        """, language="text")
        
        st.subheader("Diagnóstico Formal de Errores (Andrew Ng)")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Error en Entrenamiento (Train)", "23.45%", delta="-17.4% vs Baseline")
            st.metric("Error en Validación (Dev)", "39.70%", delta="-21.4% vs Baseline")
        with col_m2:
            st.metric("Sesgo Evitable (Train - Bayes)", "18.45%", delta="DOMINANTE", delta_color="inverse")
            st.metric("Brecha de Varianza (Dev - Train)", "16.25%", delta="Controlada", delta_color="normal")
            
        st.error("""
        **DIAGNÓSTICO FORMAL: SESGO ALTO ESTRUCTURAL DOMINANTE (UNDERFITTING)**  
        * **Causa:** El MLP regularizado superó al Baseline (Dev Acc saltó de 36.7% a 60.3%), pero chocó contra el techo del aplanamiento (12.288 números sin topología 2D).  
        * **Acción de Cátedra:** Pasar a la **Etapa 3: Redes Convolucionales (CNN)** para incorporar campos receptivos locales e invarianza translacional.
        """)

    with tab_mlp_infer:
        st.subheader("Inferencia Comparativa: Baseline Lineal vs. MLP PyTorch")
        if "current_img" in st.session_state and baseline_artifact is not None:
            arr_img = np.array(st.session_state["current_img"].resize((64, 64)))
            c_comp, c_r, c_g, c_b = st.columns(4)
            c_comp.image(st.session_state["current_img"].resize((150, 150), Image.NEAREST), caption="Parche Seleccionado", width=140)
            c_r.image(arr_img[:, :, 0], caption="Canal Rojo", clamp=True, width=140)
            c_g.image(arr_img[:, :, 1], caption="Canal Verde", clamp=True, width=140)
            c_b.image(arr_img[:, :, 2], caption="Canal Azul", clamp=True, width=140)
            
            flat = arr_img.astype(np.float32).flatten().reshape(1, -1)
            scaled = baseline_artifact["scaler"].transform(flat)
            probs_b = baseline_artifact["model"].predict_proba(scaled)[0]
            classes = baseline_artifact["classes"]
            
            probs_m = None
            if mlp_model is not None:
                with torch.no_grad():
                    t_in = torch.tensor(scaled, dtype=torch.float32).to(DEVICE)
                    probs_m = F.softmax(mlp_model(t_in), dim=1).cpu().numpy()[0]
                    
            c1, c2 = st.columns(2)
            with c1:
                idx_b = int(np.argmax(probs_b))
                st.info(f"**Baseline Lineal:** {CLASS_NAMES_ES[classes[idx_b]]} ({probs_b[idx_b]*100:.1f}%)")
            with c2:
                if probs_m is not None:
                    idx_m = int(np.argmax(probs_m))
                    st.success(f"**MLP Regularizado:** {CLASS_NAMES_ES[classes[idx_m]]} ({probs_m[idx_m]*100:.1f}%)")
                else:
                    st.warning("Checkpoint PyTorch del MLP aún no guardado.")
        else:
            st.info("👈 Primero cargá una muestra en la pestaña de Inferencia de la Etapa 1 para probar la comparativa.")

# ==============================================================================
# ETAPA 3: ROADMAP CONVOLUCIONAL (CNN)
# ==============================================================================
elif etapa == "🔬 Etapa 3: Roadmap Convolucional (CNN)":
    st.subheader("🔬 Etapa 3: Redes Neuronales Convolucionales (Consigna Oficial 4.3.A)")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("### 1. Bloques Convolucionales Jerárquicos")
        st.markdown("""
        * **Bloque 1:** `Conv2D(3 -> 32, kernel=3, padding='same')` + `BatchNorm2d` + `ReLU` + `MaxPool2D(2, 2)` $\\to [32, 32, 32]$
        * **Bloque 2:** `Conv2D(32 -> 64, kernel=3, padding='same')` + `BatchNorm2d` + `ReLU` + `MaxPool2D(2, 2)` $\\to [64, 16, 16]$
        * **Bloque 3:** `Conv2D(64 -> 128, kernel=3, padding='same')` + `BatchNorm2d` + `ReLU` + `MaxPool2D(2, 2)` $\\to [128, 8, 8]$
        * **Head Densa:** `GlobalAveragePooling2D` + `Dropout(0.40)` + `Linear(128 -> 10)`
        """)
    with col_c2:
        st.markdown("### 2. Comparación y Atribución de Errores")
        st.markdown("""
        * **End-to-End puro vs. Pipeline Secuencial:** Entrenar la CNN completa versus usar la base convolucional congelada como extractor de características + clasificador tabular.
        * **Atribución Cuantitativa de Errores:** ¿Qué porcentaje de las confusiones proviene de la etapa convolucional (texturas ambiguas) vs. la etapa de clasificación lineal?
        * **Data Augmentation:** Flips horizontales/verticales y rotaciones de 90° simulando el paso orbital del satélite Sentinel-2.
        """)

# ==============================================================================
# MLOPS LEDGER & HISTÓRICO DE EXPERIMENTOS
# ==============================================================================
elif etapa == "📊 MLOps Ledger & Experimentos":
    st.subheader("📊 MLOps Experiment Ledger & Matriz de Trade-Off")
    st.markdown("Registro inmutable de corridas generado por `src/training/tracker.py`.")
    
    tab_runs, tab_tradeoff = st.tabs([
        "📋 Bitácora de Corridas (Ledger)",
        "⚖️ Matriz de Trade-off (F1 vs. Latencia)"
    ])
    
    with tab_runs:
        if ledger_data:
            df_l = pd.DataFrame([
                {
                    "Run ID": e.get("run_id"),
                    "Fecha": e.get("timestamp"),
                    "Modelo": e.get("model_name"),
                    "Train Acc": f"{e['metrics'].get('train_accuracy', 0):.2f}%",
                    "Dev Acc": f"{e['metrics'].get('dev_accuracy', 0):.2f}%",
                    "Macro F1": f"{e['metrics'].get('macro_f1', 0):.4f}",
                    "Gap (%)": f"{e['metrics'].get('gap', 0):.2f}%",
                    "Latencia (ms)": f"{e['metrics'].get('latency_ms', 0):.2f}",
                    "Diagnóstico Clínico": e.get("diagnosis")
                }
                for e in ledger_data
            ])
            st.dataframe(df_l, use_container_width=True)
        else:
            st.info("Aún no hay registros en `experiments_ledger.json`. Corré el bloque del tracker en el notebook para poblarlos.")

    with tab_tradeoff:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            df_trade = pd.DataFrame({
                "Modelo": ["Baseline (C=0.01)", "MLP Vanilla", "MLP Regularizado", "CNN (Proyectado)"],
                "Macro_F1": [0.3524, 0.4310, 0.6030, 0.9200],
                "Latencia_ms": [32.10, 6.80, 7.15, 14.50],
                "Parametros": [122890, 6459274, 6459274, 1850000]
            })
            fig_trade = px.scatter(
                df_trade, x="Latencia_ms", y="Macro_F1", text="Modelo",
                size="Parametros", color="Modelo",
                title="Trade-off: F1-Score vs. Latencia (CPU)",
                range_y=[0.2, 1.0], range_x=[0, 40]
            )
            fig_trade.update_traces(textposition='top right')
            st.plotly_chart(fig_trade, use_container_width=True)
            
        with col_t2:
            df_g = pd.DataFrame({
                "Modelo": ["Baseline Default", "Baseline Calibrado", "MLP Vanilla", "MLP Regularizado"],
                "Brecha_Varianza": [67.10, 4.15, 51.40, 16.25]
            })
            fig_g = px.bar(
                df_g, x="Modelo", y="Brecha_Varianza",
                title="Evolución de la Brecha de Varianza (Train - Dev %)",
                color="Brecha_Varianza", color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_g, use_container_width=True)