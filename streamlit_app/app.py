import os
import re
import sys
import json
import base64
import random
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

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
    CLASS_NAMES, CLASS_NAMES_ES
)

# --- Corrida oficial entregada (Google Colab, 21/11/2026) ---
TP_DIR = os.path.join(BASE_DIR, "TP_IA2026_BARRERAS_PORCHIA_PAAL_GIMENEZ_EuroSAT")
NOTEBOOK_PATH = os.path.join(TP_DIR, "TP_IA2026_BARRERAS_PORCHIA_PAAL_GIMENEZ_GRUPO5.ipynb")
COLAB_DIR = os.path.join(BASE_DIR, "artifacts_colab", "artifacts-20261125T092450Z-1-001", "artifacts")
FIGURES_DIR = os.path.join(COLAB_DIR, "figures")
RESULTS_PATH = os.path.join(COLAB_DIR, "metrics", "etapa5_resultados.json")
LEDGER_PATH = os.path.join(COLAB_DIR, "metrics", "experiments_ledger.json")
CNN_PATH = os.path.join(COLAB_DIR, "models", "cnn_best.pt")
MLP_PATH = os.path.join(COLAB_DIR, "models", "mlp_regularized.pt")
VAE_PATH = os.path.join(COLAB_DIR, "models", "vae_eurosat.pt")
# StandardScaler del MLP: el notebook no lo guarda; se reconstruye con scripts/build_mlp_scaler.py
MLP_SCALER_PATH = os.path.join(ARTIFACTS_DIR, "models", "mlp_standard_scaler.npz")

# --- Artefactos previos (octubre), conservados para la inferencia del Baseline y la auditoría preliminar ---
BASELINE_PATH = os.path.join(ARTIFACTS_DIR, "models", "baseline_logreg.joblib")
ERRORS_METADATA_PATH = os.path.join(ARTIFACTS_DIR, "error_samples", "errors_50_metadata.json")
ERRORS_IMG_DIR = os.path.join(ARTIFACTS_DIR, "error_samples")

# Normalización de la CNN en el notebook (sección 4.3.1, estadísticos de ImageNet)
MEAN_RGB = [0.485, 0.456, 0.406]
STD_RGB = [0.229, 0.224, 0.225]

ETAPAS = [
    "🛰️ Etapa 1: Baseline Lineal & EDA",
    "🧠 Etapa 2: Red Densa (MLP) & Auditoría",
    "🔬 Etapa 3: Redes Convolucionales (CNN)",
    "🎨 Etapa 4: Modelo Generativo (VAE)",
    "🏁 Etapa 5: Evaluación Final & Ética",
    "📊 MLOps Ledger & Experimentos",
    "📓 TP Entregado (Notebook)",
]


# ==========================================
# 1. CARGA CACHEADA DE RECURSOS
# ==========================================
def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_notebook_cells():
    with open(NOTEBOOK_PATH, encoding="utf-8") as f:
        return json.load(f)["cells"]


def notebook_class(cells, class_name, stop_marker):
    """Ejecuta la definición de una clase tal como está en el notebook entregado (sin copiar código)."""
    src = next("".join(c["source"]) for c in cells
               if c["cell_type"] == "code" and f"class {class_name}" in "".join(c["source"]))
    src = src[src.index(f"class {class_name}"):]
    src = src[: src.index(stop_marker)] if stop_marker in src else src
    scope = {"nn": nn, "torch": torch}
    exec(src, scope)
    return scope[class_name]


@st.cache_resource
def load_resources():
    from torchvision.datasets import ImageFolder

    cells = load_notebook_cells()
    baseline_artifact = joblib.load(BASELINE_PATH) if os.path.exists(BASELINE_PATH) else None
    dataset = ImageFolder(root=DATA_RAW_DIR) if os.path.exists(DATA_RAW_DIR) else None
    splits = read_json(SPLITS_PATH)

    cnn_model = None
    if os.path.exists(CNN_PATH):
        cnn_model = notebook_class(cells, "EuroSATCNN", "# Inspección")(num_classes=10)
        cnn_model.load_state_dict(torch.load(CNN_PATH, map_location="cpu"))
        cnn_model.eval()

    mlp_model, mlp_scaler = None, None
    if os.path.exists(MLP_PATH) and os.path.exists(MLP_SCALER_PATH):
        mlp_model = notebook_class(cells, "MLPRegularized", "dummy =")()
        mlp_model.load_state_dict(torch.load(MLP_PATH, map_location="cpu"))
        mlp_model.eval()
        mlp_scaler = dict(np.load(MLP_SCALER_PATH))

    results = read_json(RESULTS_PATH)
    ledger_data = read_json(LEDGER_PATH, [])
    errors_metadata = read_json(ERRORS_METADATA_PATH, [])
    return baseline_artifact, dataset, splits, cnn_model, mlp_model, mlp_scaler, results, ledger_data, errors_metadata


baseline_artifact, dataset, splits, cnn_model, mlp_model, mlp_scaler, results, ledger_data, errors_metadata = load_resources()
NB_CELLS = load_notebook_cells()
MODELS = {m["Modelo"]: m for m in results["models"]}


# ==========================================
# 2. UTILIDADES: NOTEBOOK ENTREGADO, FIGURAS Y LOGS
# ==========================================
def figure(n, caption=None):
    """Figura exportada por el notebook. Las casi cuadradas (matrices de confusión) se centran con un ancho
    que entra en pantalla sin scroll; el botón de pantalla completa aparece al pasar el mouse."""
    path = os.path.join(FIGURES_DIR, f"fig_{n:03d}.png")
    w, h = Image.open(path).size
    box = st.columns([1, 3, 1])[1] if h / w > 0.75 else st
    box.image(path, caption=caption, width="stretch")


def render_output(out):
    if out["output_type"] == "stream":
        if out.get("name") == "stderr":
            return
        st.code("".join(out["text"]), language=None)
        return
    data = out.get("data", {})
    if "image/png" in data:
        st.image(base64.b64decode(data["image/png"]), width="stretch")
    elif "text/html" in data and (m := re.search(r"<table.*?</table>", "".join(data["text/html"]), re.S)):
        st.markdown(f'<div class="nb-table">{m.group(0)}</div>', unsafe_allow_html=True)
    elif "text/markdown" in data:
        st.markdown("".join(data["text/markdown"]), unsafe_allow_html=True)
    elif "text/plain" in data:
        st.code("".join(data["text/plain"]), language=None)


def nb(*idxs, show_code=False):
    """Muestra celdas del notebook entregado: markdown y salidas (el código sólo si se pide)."""
    for i in idxs:
        cell = NB_CELLS[i]
        src = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            st.markdown(src, unsafe_allow_html=True)
            continue
        if show_code and src.strip():
            with st.expander("Código de la celda", expanded=False):
                st.code(src, language="python")
        for out in cell.get("outputs", []):
            render_output(out)


@st.cache_data
def training_logs():
    """Curvas de entrenamiento de la corrida del 21/11, tomadas de lo que imprimió el notebook."""
    sup = re.compile(r"\[(.+?)\] Época (\d+)/\d+ \| Train Loss: ([\d.]+) - Dev Loss: ([\d.]+) \| "
                     r"Train F1: ([\d.]+) - Dev F1: ([\d.]+)")
    vae = re.compile(r"Época \[(\d+)/\d+\] \| Train: ([\d.]+) \(BCE: ([\d.]+), KL: ([\d.]+)\) \| "
                     r"Dev: ([\d.]+) \(BCE: ([\d.]+), KL: ([\d.]+)\)")
    rows, vrows = [], []
    for cell in load_notebook_cells():
        for out in cell.get("outputs", []):
            if out["output_type"] != "stream":
                continue
            for line in "".join(out["text"]).splitlines():
                if m := sup.search(line):
                    rows.append([m[1], int(m[2])] + [float(x) for x in m.groups()[2:]])
                elif m := vae.search(line):
                    vrows.append([int(m[1])] + [float(x) for x in m.groups()[1:]])
    df = pd.DataFrame(rows, columns=["Modelo", "Época", "Train Loss", "Dev Loss", "Train F1", "Dev F1"])
    dv = pd.DataFrame(vrows, columns=["Época", "Train Total", "Train BCE", "Train KL", "Dev Total", "Dev BCE", "Dev KL"])
    return df, dv


def curves_chart(df, models, metric_pair, title, yaxis):
    fig = go.Figure()
    palette = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b"]
    for k, name in enumerate(models):
        d = df[df["Modelo"] == name]
        for metric, dash in zip(metric_pair, ["dot", "solid"]):
            fig.add_trace(go.Scatter(x=d["Época"], y=d[metric], mode="lines+markers", name=f"{name}: {metric}",
                                     line=dict(color=palette[k % len(palette)], width=2, dash=dash)))
    fig.update_layout(title=title, xaxis_title="Época", yaxis_title=yaxis, height=360,
                      margin=dict(l=0, r=0, t=35, b=10), legend=dict(font=dict(size=10)))
    return fig


def confusion_heatmap(cm, normalized, colorscale="Blues", height=600):
    cm = np.array(cm)
    classes_es = [CLASS_NAMES_ES[c] for c in CLASS_NAMES]
    if normalized:
        z = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        text = [[f"{v:.2f}" for v in row] for row in z]
    else:
        z = cm
        text = [[str(v) for v in row] for row in z]
    fig = px.imshow(z, x=classes_es, y=classes_es, color_continuous_scale=colorscale,
                    labels=dict(x="Predicción del Modelo", y="Clase Real (Ground Truth)", color="Valor"))
    fig.update_traces(text=text, texttemplate="%{text}", textfont=dict(size=11))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=20, b=0))
    return fig


# ==========================================
# 3. INFERENCIA EN VIVO (Baseline, MLP y CNN)
# ==========================================
def predict_baseline(img):
    flat = np.array(img.resize((64, 64))).astype(np.float32).flatten().reshape(1, -1)
    probs = baseline_artifact["model"].predict_proba(baseline_artifact["scaler"].transform(flat))[0]
    order = [list(baseline_artifact["classes"]).index(c) for c in CLASS_NAMES]
    return probs[order]


def predict_mlp(img):
    flat = np.array(img.resize((64, 64))).astype(np.float32).flatten()
    x = (flat - mlp_scaler["mean"]) / mlp_scaler["scale"]
    with torch.no_grad():
        return F.softmax(mlp_model(torch.tensor(x, dtype=torch.float32).unsqueeze(0)), dim=1)[0].numpy()


def predict_cnn(img):
    tf = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(MEAN_RGB, STD_RGB)])
    with torch.no_grad():
        return F.softmax(cnn_model(tf(img).unsqueeze(0)), dim=1)[0].numpy()


# Desplazamiento (px) de cada etiqueta respecto de su punto, para que no se superpongan
LABEL_OFFSETS = {
    "Secuencial B (backbone propio + LogReg)": (95, -40), "Secuencial A (ResNet-18 + LogReg)": (95, 35),
    "CNN sin augmentation": (-90, -40), "CNN con augmentation": (100, 20),
    "CNN con augmentation (4.000 imágenes)": (110, 35),
    "MLP Regularizado": (-70, -35), "MLP Vanilla": (-50, 40), "VAE (d=64) + linear probe": (-110, 30),
    "Baseline LogReg (C=0.01)": (110, -20), "Baseline LogReg (C=1.0)": (110, 20),
}
ETAPA_COLORS = {1: "#f59e0b", 2: "#3b82f6", 3: "#10b981", 4: "#ec4899"}


def complexity_chart(df_models, title, size_by_time=False):
    """Dev Macro F1 vs. parámetros, con etiquetas desplazadas. Con size_by_time el tamaño del punto = tiempo de entrenamiento."""
    df = df_models.dropna(subset=["Parámetros", "Dev Macro F1"])
    t_max = df["Tiempo Entren. (s)"].max()
    fig = go.Figure()
    for etapa_n, d in df.groupby("Etapa"):
        size = (10 + 35 * np.sqrt(d["Tiempo Entren. (s)"] / t_max)) if size_by_time else 13
        fig.add_trace(go.Scatter(x=d["Parámetros"], y=d["Dev Macro F1"], mode="markers", name=f"Etapa {etapa_n}",
                                 marker=dict(size=size, color=ETAPA_COLORS.get(etapa_n), opacity=0.85,
                                             line=dict(width=1, color="white")),
                                 customdata=np.stack([d["Modelo"], d["Tiempo Entren. (s)"]], axis=-1),
                                 hovertemplate="<b>%{customdata[0]}</b><br>Parámetros: %{x:,}<br>Dev Macro F1: %{y:.4f}"
                                               "<br>Tiempo de entrenamiento: %{customdata[1]:.0f} s<extra></extra>"))
    for _, r in df.iterrows():
        ax, ay = LABEL_OFFSETS.get(r["Modelo"], (0, -30))
        fig.add_annotation(x=np.log10(r["Parámetros"]), y=r["Dev Macro F1"], text=f"{r['Modelo']} ({r['Dev Macro F1']:.3f})",
                           ax=ax, ay=ay, showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="#9ca3af",
                           font=dict(size=12), bgcolor="rgba(30,30,40,0.85)", bordercolor="#4b5563", borderpad=3)
    fig.update_layout(title=title, height=600, margin=dict(l=0, r=0, t=40, b=10),
                      xaxis=dict(type="log", title="Parámetros entrenables (escala log)", range=[3.3, 7.3]),
                      yaxis=dict(title="Dev Macro F1", range=[0.22, 1.02]),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=1, xanchor="right"))
    return fig


@st.cache_data
def geometry_demo():
    """Caso sintético 3D: una clase encerrada dentro de otra. Entrena un modelo lineal y un MLP sobre los mismos puntos."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    rng = np.random.default_rng(0)

    def shell(n, rmin, rmax):
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v * rng.uniform(rmin, rmax, size=(n, 1))

    X = np.vstack([shell(500, 0, 1.1), shell(500, 1.6, 2.4)])
    y = np.r_[np.zeros(500), np.ones(500)]
    lin = LogisticRegression().fit(X, y)
    mlp = MLPClassifier((32, 16), max_iter=2000, random_state=0).fit(X, y)
    g = np.linspace(-2.6, 2.6, 28)
    G = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    return X, y, lin.coef_[0], lin.intercept_[0], lin.score(X, y), mlp.score(X, y), G, mlp.predict_proba(G)[:, 1]


SAMPLES_DIR = os.path.join(BASE_DIR, "streamlit_app", "samples")  # generado por scripts/export_demo_samples.py


def random_sample(split, class_name=None):
    """Devuelve (imagen, clase real en español) al azar del split pedido.
    Usa el dataset crudo si está disponible; si no (p. ej. en la nube), la muestra exportada en streamlit_app/samples/."""
    if dataset is not None and splits is not None:
        pool = splits[f"{split}_indices"]
        if class_name:
            target = CLASS_NAMES.index(class_name)
            pool = [i for i in pool if dataset.targets[i] == target]
        img, target = dataset[random.choice(pool)]
        return img, CLASS_NAMES_ES[CLASS_NAMES[target]]
    cls = class_name or random.choice(CLASS_NAMES)
    folder = os.path.join(SAMPLES_DIR, split, cls)
    path = os.path.join(folder, random.choice(sorted(os.listdir(folder))))
    return Image.open(path).convert("RGB"), CLASS_NAMES_ES[cls]


def image_selector(key):
    """Elegir una muestra al azar (Dev o Test) o subir una imagen propia. Guarda la selección en la sesión."""
    modos = ["Muestra del Dev Set", "Muestra del Test Set", "Subir Imagen Propia"]
    modo = st.radio("Método de Selección de Imagen:", modos, horizontal=True, key=f"{key}_modo")
    if modo != modos[2]:
        split = "dev" if modo == modos[0] else "test"
        opciones = ["🎲 Cualquier clase"] + list(CLASS_NAMES_ES.values())
        clase_elegida = st.selectbox("Selecciona Cobertura a Probar:", opciones, key=f"{key}_clase")
        if st.button("🎲 Cargar Muestra Aleatoria", width="stretch", key=f"{key}_btn"):
            class_name = None if clase_elegida == opciones[0] else \
                [k for k, v in CLASS_NAMES_ES.items() if v == clase_elegida][0]
            img, gt = random_sample(split, class_name)
            st.session_state["current_img"] = img
            st.session_state["ground_truth"] = gt
            st.session_state["img_origin"] = "Dev" if split == "dev" else "Test"
        if split == "test":
            st.caption("Test (2.700 imágenes) no se usó para entrenar ni para elegir modelos: es una prueba justa.")
        if "current_img" in st.session_state:
            st.info(f"Ground Truth (Clase Real): **{st.session_state.get('ground_truth', '')}**"
                    f" — {st.session_state.get('img_origin', '')}")
    else:
        up_file = st.file_uploader("Sube un parche (.png, .jpg)", type=["png", "jpg", "jpeg", "tif", "tiff"], key=f"{key}_up")
        if up_file:
            st.session_state["current_img"] = Image.open(up_file).convert("RGB")
            st.session_state["ground_truth"] = "Subida por Usuario"
            st.session_state["img_origin"] = ""
    return st.session_state.get("current_img")


def show_channels(img):
    arr_img = np.array(img.resize((64, 64)))
    c_orig, c_r, c_g, c_b = st.columns(4)
    c_orig.image(img.resize((150, 150), Image.NEAREST), caption="Parche Compuesto", width=140)
    c_r.image(arr_img[:, :, 0], caption="Canal Rojo (R)", clamp=True, width=140)
    c_g.image(arr_img[:, :, 1], caption="Canal Verde (G)", clamp=True, width=140)
    c_b.image(arr_img[:, :, 2], caption="Canal Azul (B)", clamp=True, width=140)


def verdict(name, probs, box):
    idx = int(np.argmax(probs))
    pred_es = CLASS_NAMES_ES[CLASS_NAMES[idx]]
    gt = st.session_state.get("ground_truth")
    mark = "" if gt in (None, "Subida por Usuario") else (" ✅" if pred_es == gt else " ❌")
    box(f"**{name}:** {pred_es} ({probs[idx] * 100:.1f}%){mark}")


# ==========================================
# 4. SIDEBAR: NAVEGACIÓN EXCLUSIVA POR ETAPAS
# ==========================================
st.markdown(
    """<style>
    .nb-table { overflow-x: auto; margin-bottom: 1rem; }
    .nb-table table { border-collapse: collapse; font-size: 0.85rem; }
    .nb-table th, .nb-table td { padding: 4px 10px; border: 1px solid rgba(128,128,128,0.3); }
    </style>""",
    unsafe_allow_html=True,
)

st.sidebar.title("🛰️ EuroSAT AI Lab")
st.sidebar.markdown("**Consola Científica & MLOps**")
st.sidebar.caption("Inteligencia Artificial (UNaB 2026) | **Lic. Pablo Moreira** | **Grupo 5**")
st.sidebar.divider()

etapa = st.sidebar.radio("Navegación del Trabajo Práctico:", ETAPAS)

st.sidebar.divider()
st.sidebar.subheader("Telemetría de Artefactos")
st.sidebar.caption("Corrida oficial: Google Colab (GPU), 21/11/2026")
st.sidebar.markdown(f"• **EuroSAT Raw:** {'✅ Conectado' if dataset else '⚠️ Ausente (muestras de respaldo)'}")
st.sidebar.markdown(f"• **Particiones (80/10/10):** {'✅ Generadas' if splits else '❌ No halladas'}")
st.sidebar.markdown(f"• **Notebook entregado:** ✅ {len(NB_CELLS)} celdas")
st.sidebar.markdown(f"• **Baseline (LogReg):** {'✅ Serializado (oct.)' if baseline_artifact else '⚠️ Pendiente'}")
st.sidebar.markdown(f"• **MLP PyTorch:** {'✅ Checkpoint 21/11' if mlp_model else '⚠️ Falta checkpoint o scaler'}")
st.sidebar.markdown(f"• **CNN Campeona:** {'✅ Checkpoint 21/11' if cnn_model else '⚠️ No hallada'}")
st.sidebar.markdown(f"• **VAE:** {'✅ Checkpoint 21/11' if os.path.exists(VAE_PATH) else '⚠️ No hallado'}")
st.sidebar.markdown(f"• **Experimentos (Ledger):** ✅ {len(ledger_data)} corridas")

# Header Global
st.title("🛰️ EuroSAT AI Lab — Consola Científica & MLOps")
st.markdown("**Asignatura:** Inteligencia Artificial (UNaB 2026) | **Docente:** Lic. Pablo Moreira | **Grupo 5**")
st.divider()

# ==============================================================================
# ETAPA 1: BASELINE LINEAL & EDA
# ==============================================================================
if etapa == ETAPAS[0]:
    tab_kpi, tab_3d, tab_cm, tab_inference = st.tabs([
        "📈 Diagnóstico Clínico & KPIs",
        "🧊 Espacio Espectral 3D (EDA)",
        "🎯 Matriz de Confusión",
        "🔍 Inferencia en Vivo & Canales"
    ])

    b1, b2 = MODELS["Baseline LogReg (C=1.0)"], MODELS["Baseline LogReg (C=0.01)"]
    bayes = 5.0
    train_err, dev_err = 100 - b2["Train Acc (%)"], 100 - b2["Dev Acc (%)"]
    sesgo, varianza = train_err - bayes, dev_err - train_err

    with tab_kpi:
        st.subheader("Evaluación Global del Baseline Calibrado (Dev Set)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Métrica de Optimización (Macro F1)", f"{b2['Dev Macro F1']:.4f}", delta="Referencia Etapa 1", delta_color="off")
        col2.metric("Exactitud en Validación (Dev Acc)", f"{b2['Dev Acc (%)']:.2f}%")
        col3.metric("Exactitud en Entrenamiento (Train Acc)", f"{b2['Train Acc (%)']:.2f}%")
        col4.metric("Parámetros (P/N ≈ 30,7)", f"{int(b2['Parámetros']):,}".replace(",", "."))
        st.markdown("---")
        st.subheader("Descomposición Formal del Error (Metodología Andrew Ng - Clase 3)")

        fig_error = go.Figure()
        fig_error.add_trace(go.Bar(y=["Descomposición del Error"], x=[bayes], name=f"Error Humano / Bayes ({bayes:.1f}%)",
                                   orientation='h', marker=dict(color="#10b981")))
        fig_error.add_trace(go.Bar(y=["Descomposición del Error"], x=[sesgo], name=f"Sesgo Evitable ({sesgo:.2f}%)",
                                   orientation='h', marker=dict(color="#3b82f6")))
        fig_error.add_trace(go.Bar(y=["Descomposición del Error"], x=[varianza], name=f"Brecha de Varianza ({varianza:.2f}% - DOMINANTE)",
                                   orientation='h', marker=dict(color="#ef4444")))
        fig_error.update_layout(barmode='stack', height=200, margin=dict(l=0, r=0, t=50, b=10),
                                xaxis=dict(title="Porcentaje de Error Acumulado (%)", range=[0, 100]),
                                legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0))
        st.plotly_chart(fig_error, width="stretch")
        st.error(f"""
        **DIAGNÓSTICO FORMAL: ALTA VARIANZA (OVERFITTING)**
        * **Causa:** La brecha de varianza (${varianza:.2f}\\%$) domina sobre el sesgo evitable (${sesgo:.2f}\\%$). En el régimen $P \\gg N$ (122.890 parámetros para 4.000 imágenes) el modelo memoriza Train ({b2['Train Acc (%)']:.1f}%) y generaliza mal en Dev ({b2['Dev Acc (%)']:.1f}%).
        * **Regularización $L_2$:** con $C=0.01$ la brecha baja de {b1['Brecha Train-Dev (pp)']:.1f} a {b2['Brecha Train-Dev (pp)']:.1f} pp, pero un hiperplano sobre 12.288 píxeles aplanados no alcanza.
        * **Acción para la Etapa 2:** auditar 50 errores en Dev e implementar un **Perceptrón Multicapa (MLP)** con activaciones no lineales (GELU) y regularización.
        """)

        st.markdown("#### Experimentos del Baseline (corrida del 21/11)")
        df_b = pd.DataFrame([b1, b2])[["Modelo", "Configuración", "Train Acc (%)", "Dev Acc (%)", "Dev Macro F1",
                                       "Brecha Train-Dev (pp)", "Test Acc (%)", "Test Macro F1"]]
        st.dataframe(df_b, width="stretch", hide_index=True)
        with st.expander("📄 Conclusiones Oficiales de la Etapa 1 (notebook entregado)"):
            nb(27, 28)

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
                    scene=dict(xaxis_title="Canal Rojo (R)", yaxis_title="Canal Verde (G)", zaxis_title="Canal Azul (B)"),
                    height=650, margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_3d, width="stretch")
                st.info("""
                **Lectura Científica del Gráfico 3D:**
                * **Mar o Lago:** Agrupada abajo a la izquierda en zonas de baja reflectancia ($R, G, B < 60$), fácilmente separable.
                * **Bosque, Pastizal y Cultivos:** Forman un racimo superpuesto a lo largo del eje Verde, demostrando por qué una frontera plana genera confusión cruzada.
                """)
        else:
            st.info("El cubo 3D se calcula sobre las imágenes crudas de EuroSAT, que no están disponibles en este entorno.")
        with st.expander("📊 Visualizaciones del EDA (notebook entregado)"):
            figure(1)
            figure(2)
            figure(3)

    with tab_cm:
        st.subheader("Matriz de Confusión Oficial del Baseline Calibrado (Dev Set)")
        tipo_cm = st.radio("Modalidad de Visualización:", ["Recall Normalizado (Tasa de Acierto)", "Conteos Absolutos (Muestras Reales)"], horizontal=True)
        figure(5 if "Normalizado" in tipo_cm else 4, caption="Corrida del 21/11 — Baseline LogReg (C=0.01), 1.000 imágenes de Dev")

    with tab_inference:
        st.subheader("Inferencia Satelital en Tiempo Real")
        col_input, col_view = st.columns([1, 2])
        with col_input:
            img_to_infer = image_selector("e1")
        with col_view:
            if img_to_infer is not None and baseline_artifact is not None:
                show_channels(img_to_infer)
                probs = predict_baseline(img_to_infer)
                pred_idx = int(np.argmax(probs))
                st.success(f"**Predicción del Modelo:** {CLASS_NAMES_ES[CLASS_NAMES[pred_idx]]} ({CLASS_NAMES[pred_idx]}) — **Confianza:** {probs[pred_idx]*100:.2f}%")
                df_bar = pd.DataFrame({"Clase": [CLASS_NAMES_ES[c] for c in CLASS_NAMES],
                                       "Probabilidad (%)": probs * 100}).sort_values("Probabilidad (%)", ascending=True)
                fig_bar = px.bar(df_bar, x="Probabilidad (%)", y="Clase", orientation='h',
                                 text=df_bar["Probabilidad (%)"].apply(lambda x: f"{x:.1f}%"),
                                 color="Probabilidad (%)", color_continuous_scale="Viridis", range_x=[0, 100])
                fig_bar.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=10))
                st.plotly_chart(fig_bar, width="stretch")
                st.caption("ℹ️ El Baseline serializado (`baseline_logreg.joblib`) proviene del script de entrenamiento de octubre: el notebook entregado no exporta el modelo lineal. Su Macro F1 en Test es 0,3514 (notebook: 0,3515).")

# ==============================================================================
# ETAPA 2: RED DENSA (MLP) & AUDITORÍA DE ERRORES
# ==============================================================================
elif etapa == ETAPAS[1]:
    tab_audit, tab_arch, tab_mlp_infer = st.tabs([
        "🔬 Auditoría de 50 Errores del Baseline",
        "🏗️ Arquitectura MLP & Curvas",
        "🔍 Inferencia MLP en Vivo"
    ])

    with tab_audit:
        st.subheader("🔬 Auditoría Clínica de 50 Muestras de Error (Sección 4.2.1)")
        st.write("Análisis sistemático de fallas en el Dev Set para fundamentar las mejoras arquitectónicas hacia el MLP y CNN.")
        nb(33)
        figure(6, caption="Galería de los 50 errores auditados (corrida del 21/11)")
        nb(35)
        st.markdown("#### Discusión de Hallazgos y Propuestas de Mejora")
        nb(37)

        with st.expander("🗂️ Análisis manual preliminar (octubre) — tarjetas de auditoría"):
            if errors_metadata:
                st.caption("Auditoría previa a la corrida oficial, con otra muestra de 50 errores y otra taxonomía. Se conserva como antecedente del análisis.")
                df_err = pd.DataFrame(errors_metadata)
                col_f1, col_f2 = st.columns([2, 1])
                with col_f1:
                    categorias = ["Todas"] + list(df_err["error_category"].unique())
                    cat_sel = st.selectbox("Filtrar por Causa Diagnóstica:", categorias)
                with col_f2:
                    st.metric("Total de Muestras Auditadas", f"{len(df_err)} casos", delta="Dev Set")

                filtered_err = df_err if cat_sel == "Todas" else df_err[df_err["error_category"] == cat_sel]
                st.caption(f"Mostrando **{len(filtered_err)}** casos de error correspondientes:")

                cols = st.columns(2)
                for idx, (_, r) in enumerate(filtered_err.iterrows()):
                    col = cols[idx % 2]
                    img_p = os.path.join(ERRORS_IMG_DIR, r["image_file"])
                    with col:
                        with st.container(border=True):
                            c_hdr1, c_hdr2 = st.columns([2, 1])
                            c_hdr1.markdown(f"**Caso #{r['error_id']:02d}**")
                            c_hdr2.markdown(
                                f"<p style='text-align: right; margin: 0; font-size: 0.85rem; font-weight: 600; color: #ef4444;'>"
                                f"Conf: {r['confidence_predicted']}%</p>",
                                unsafe_allow_html=True
                            )
                            c_img, c_meta = st.columns([1, 2.5])
                            with c_img:
                                if os.path.exists(img_p):
                                    st.image(Image.open(img_p), width=96)
                                st.caption("64×64 • RGB")
                            with c_meta:
                                st.markdown(f"🎯 **Real:** `{r['real_class_es']}`")
                                st.markdown(f"❌ **Predicho:** :red[**{r['predicted_class_es']}**]")
                                st.caption(f"🏷️ *{r['error_category']}*")
                            st.markdown(f"> 💡 **Diagnóstico:** {r['clinical_justification']}")
            else:
                st.warning("No se encontró `errors_50_metadata.json`.")

    with tab_arch:
        st.subheader("🏗️ Arquitectura MLP, Curvas de Aprendizaje y Fronteras No Lineales")
        v, r = MODELS["MLP Vanilla"], MODELS["MLP Regularizado"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dev Macro F1 (Regularizado)", f"{r['Dev Macro F1']:.4f}",
                  delta=f"+{(r['Dev Macro F1'] - MODELS['Baseline LogReg (C=0.01)']['Dev Macro F1']) * 100:.2f} pp vs Baseline")
        c2.metric("Dev Macro F1 (Vanilla)", f"{v['Dev Macro F1']:.4f}")
        c3.metric("Brecha Train-Dev", f"{r['Brecha Train-Dev (pp)']:.1f} pp",
                  delta=f"{r['Brecha Train-Dev (pp)'] - v['Brecha Train-Dev (pp)']:.1f} pp vs Vanilla", delta_color="inverse")
        c4.metric("Parámetros", f"{int(r['Parámetros']):,}".replace(",", "."))

        st.markdown("### 1. Curvas de Aprendizaje (Vanilla vs. Regularizado)")
        df_logs, _ = training_logs()
        mlps = ["MLP Vanilla", "MLP Regularizado"]
        c_c1, c_c2 = st.columns(2)
        c_c1.plotly_chart(curves_chart(df_logs, mlps, ["Train Loss", "Dev Loss"], "Pérdida (CrossEntropy)", "Loss"), width="stretch")
        c_c2.plotly_chart(curves_chart(df_logs, mlps, ["Train F1", "Dev F1"], "Macro F1 (Train vs Dev)", "Macro F1"), width="stretch")
        st.caption("🔍 **Interpretación:** el Vanilla memoriza (Dev Loss sube desde la época 5 mientras Train Loss baja); el Regularizado (BatchNorm + Dropout 0,35 + L2) se corta por Early Stopping en la época 20 y restaura los pesos con Dev F1 = 0,5996. Puntos: valores impresos cada 5 épocas por el notebook.")
        with st.expander("📈 Figura completa de las curvas (todas las épocas)"):
            figure(7)

        st.divider()
        st.markdown("### 2. Comparativa y Ganancia sobre el Baseline")
        nb(45)

        st.divider()
        st.markdown("### 3. Matriz de Confusión del MLP Regularizado (Dev Set)")
        figure(8, caption="Corrida del 21/11")

        st.divider()
        st.markdown("### 4. Geometría de Decisión: ¿por qué una red no lineal supera al Baseline?")
        st.markdown(
            "**Para qué sirve esta demo:** en la Etapa 1 el cubo RGB mostró que las coberturas forman *racimos* "
            "mezclados, unos dentro o alrededor de otros. El Baseline (regresión logística) sólo puede separar clases "
            "con un **plano**; el MLP, al encadenar capas con activaciones no lineales, puede **doblar** la frontera y "
            "envolver un racimo. Las imágenes reales viven en 12.288 dimensiones y no se pueden dibujar, así que acá "
            "se entrenan ambos tipos de modelo sobre un caso sintético en 3D: una clase encerrada dentro de otra. "
            "*Es una ilustración propia de la app, no forma parte del notebook entregado.*")
        X_g, y_g, w_lin, b_lin, acc_lin, acc_mlp, grid_pts, p_mlp = geometry_demo()

        def points_trace():
            return [go.Scatter3d(x=X_g[y_g == k, 0], y=X_g[y_g == k, 1], z=X_g[y_g == k, 2], mode="markers",
                                 marker=dict(size=2.5, color=col), name=nombre)
                    for k, col, nombre in [(0, "#f59e0b", "Clase A (racimo interno)"), (1, "#3b82f6", "Clase B (envolvente)")]]

        def scene_layout(fig, title):
            ax = dict(range=[-2.6, 2.6], showticklabels=False, title="")
            fig.update_layout(title=title, height=460, margin=dict(l=0, r=0, t=40, b=0),
                              scene=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="cube"),
                              legend=dict(orientation="h", y=0))

        c_geo1, c_geo2 = st.columns(2)
        with c_geo1:
            # Plano w·x + b = 0 despejado sobre el eje de mayor peso
            k = int(np.argmax(np.abs(w_lin)))
            o = [i for i in range(3) if i != k]
            g = np.linspace(-2.6, 2.6, 20)
            A, B = np.meshgrid(g, g)
            C = -(w_lin[o[0]] * A + w_lin[o[1]] * B + b_lin) / w_lin[k]
            coords = {o[0]: A, o[1]: B, k: np.clip(C, -2.6, 2.6)}
            fig_lin = go.Figure(points_trace() + [go.Surface(x=coords[0], y=coords[1], z=coords[2], opacity=0.45,
                                                             showscale=False, colorscale=[[0, "#ef4444"], [1, "#ef4444"]],
                                                             name="Frontera lineal")])
            scene_layout(fig_lin, f"Baseline lineal (LogReg): un plano — exactitud {acc_lin * 100:.0f}%")
            st.plotly_chart(fig_lin, width="stretch")
            st.caption("❌ Cualquier plano deja parte del racimo interno de cada lado: no puede aislarlo.")
        with c_geo2:
            fig_q = go.Figure(points_trace() + [go.Isosurface(
                x=grid_pts[:, 0], y=grid_pts[:, 1], z=grid_pts[:, 2], value=p_mlp, isomin=0.5, isomax=0.5,
                surface_count=1, opacity=0.35, showscale=False, colorscale=[[0, "#10b981"], [1, "#10b981"]],
                caps=dict(x_show=False, y_show=False, z_show=False), name="Frontera MLP")])
            scene_layout(fig_q, f"MLP (capas ocultas no lineales): superficie cerrada — exactitud {acc_mlp * 100:.0f}%")
            st.plotly_chart(fig_q, width="stretch")
            st.caption("✅ La frontera se curva hasta envolver el racimo (acá con ReLU; en el TP, GELU: mismo efecto).")
        st.info(f"**Conexión con el TP:** es la misma limitación que se ve con datos reales en la corrida del 21/11: "
                f"el Baseline lineal llega a Macro F1 {MODELS['Baseline LogReg (C=0.01)']['Dev Macro F1']:.3f} en Dev "
                f"y el MLP no lineal a {MODELS['MLP Regularizado']['Dev Macro F1']:.3f}. Podés rotar ambos cubos con el mouse.")

        with st.expander("📄 Conclusiones Oficiales de la Etapa 2 (notebook entregado)"):
            nb(48, 49)

    with tab_mlp_infer:
        st.subheader("Inferencia Comparativa: Baseline Lineal vs. MLP PyTorch")
        if "current_img" in st.session_state and baseline_artifact is not None:
            show_channels(st.session_state["current_img"])
            c1, c2 = st.columns(2)
            with c1:
                verdict("Baseline Lineal", predict_baseline(st.session_state["current_img"]), st.info)
            with c2:
                if mlp_model is not None:
                    verdict("MLP Regularizado", predict_mlp(st.session_state["current_img"]), st.success)
                else:
                    st.warning("Falta el checkpoint del MLP o su StandardScaler (`python scripts/build_mlp_scaler.py`).")
        else:
            st.info("👈 Primero cargá una muestra en la pestaña de Inferencia de la Etapa 1 (o de la Etapa 3) para probar la comparativa.")

# ==============================================================================
# ETAPA 3: REDES NEURONALES CONVOLUCIONALES (CNN)
# ==============================================================================
elif etapa == ETAPAS[2]:
    st.subheader("🔬 Etapa 3: Redes Neuronales Convolucionales (Consigna Oficial 4.3.A)")
    tab_res, tab_arch, tab_seq, tab_conf, tab_inf = st.tabs([
        "📈 Resultados & KPIs",
        "🏗️ Arquitectura & Data Augmentation",
        "🧩 End-to-End vs. Secuencial & Atribución",
        "🎯 Confusiones Resueltas",
        "🔍 Inferencia en Vivo (3 Etapas)"
    ])
    cnn = MODELS["CNN sin augmentation"]
    stage3 = [m for m in results["models"] if m["Etapa"] == 3]

    with tab_res:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mejor CNN (Dev Macro F1)", f"{cnn['Dev Macro F1']:.4f}",
                  delta=f"+{(cnn['Dev Macro F1'] - MODELS['MLP Regularizado']['Dev Macro F1']) * 100:.1f} pp vs MLP")
        c2.metric("Exactitud en Dev", f"{cnn['Dev Acc (%)']:.2f}%")
        c3.metric("Parámetros CNN", f"{int(cnn['Parámetros']):,}".replace(",", "."),
                  delta=f"{int(cnn['Parámetros']) / int(MODELS['MLP Regularizado']['Parámetros']) * 100:.1f}% del MLP", delta_color="off")
        c4.metric("Brecha Train-Dev", f"{cnn['Brecha Train-Dev (pp)']:.2f} pp")

        df_bar = pd.DataFrame([MODELS["Baseline LogReg (C=0.01)"], MODELS["MLP Regularizado"]] + stage3)
        fig = px.bar(df_bar, x="Dev Macro F1", y="Modelo", orientation="h", color="Etapa", text=df_bar["Dev Macro F1"].map("{:.4f}".format),
                     color_continuous_scale="Viridis", range_x=[0, 1], title="Dev Macro F1 por modelo (corrida del 21/11)")
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=35, b=10), yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(pd.DataFrame(stage3)[["Modelo", "Arquitectura", "Configuración", "Parámetros", "Imágenes Train",
                                           "Train Acc (%)", "Dev Acc (%)", "Dev Macro F1", "Tiempo Entren. (s)"]],
                     width="stretch", hide_index=True)
        st.markdown("#### Reporte de Ganancia sobre el Número Único (mismo subconjunto de Dev)")
        nb(82, 83)
        with st.expander("📄 Conclusiones Oficiales de la Etapa 3 (notebook entregado)"):
            nb(84)

    with tab_arch:
        nb(54)
        st.markdown("### Pipeline de Data Augmentation Específico del Dominio Satelital")
        figure(9)
        st.markdown("### Curvas de Aprendizaje de los 3 Experimentos End-to-End")
        df_logs, _ = training_logs()
        cnns = ["CNN Sin Augmentation", "CNN Con Augmentation", "CNN Con Augmentation (4.000 imágenes)"]
        c_c1, c_c2 = st.columns(2)
        c_c1.plotly_chart(curves_chart(df_logs, cnns, ["Train Loss", "Dev Loss"], "Pérdida (CrossEntropy)", "Loss"), width="stretch")
        c_c2.plotly_chart(curves_chart(df_logs, cnns, ["Train F1", "Dev F1"], "Macro F1 (Train vs Dev)", "Macro F1"), width="stretch")
        st.caption("Valores impresos por el notebook cada 3 épocas (20 épocas en GPU).")
        with st.expander("📈 Figura completa de las curvas (todas las épocas)"):
            figure(10)
        nb(66, 67)

    with tab_seq:
        nb(70, 71, 72, 73)

    with tab_conf:
        nb(75, 76, 77, 78, 79)

    with tab_inf:
        st.subheader("Inferencia Comparativa: Baseline vs. MLP vs. CNN Campeona")
        col_input, col_view = st.columns([1, 2])
        with col_input:
            img_to_infer = image_selector("e3")
        with col_view:
            if img_to_infer is not None:
                show_channels(img_to_infer)
                preds = {}
                if baseline_artifact is not None:
                    preds["Baseline Lineal (oct.)"] = predict_baseline(img_to_infer)
                if mlp_model is not None:
                    preds["MLP Regularizado"] = predict_mlp(img_to_infer)
                if cnn_model is not None:
                    preds["CNN Campeona"] = predict_cnn(img_to_infer)
                for col, (name, probs), box in zip(st.columns(len(preds)), preds.items(), [st.info, st.warning, st.success]):
                    with col:
                        verdict(name, probs, box)
                df_p = pd.DataFrame([{"Modelo": n, "Clase": CLASS_NAMES_ES[c], "Probabilidad (%)": p[i] * 100}
                                     for n, p in preds.items() for i, c in enumerate(CLASS_NAMES)])
                fig_p = px.bar(df_p, x="Probabilidad (%)", y="Clase", color="Modelo", orientation="h", barmode="group",
                               range_x=[0, 100], color_discrete_sequence=["#3b82f6", "#f59e0b", "#10b981"])
                fig_p.update_layout(height=480, margin=dict(l=0, r=0, t=10, b=10))
                st.plotly_chart(fig_p, width="stretch")

# ==============================================================================
# ETAPA 4: MODELO GENERATIVO (VAE)
# ==============================================================================
elif etapa == ETAPAS[3]:
    st.subheader("🎨 Etapa 4: Modelo Generativo — Variational Autoencoder (VAE)")
    tab_train, tab_rec, tab_lat, tab_q, tab_conc = st.tabs([
        "📈 Diseño & Entrenamiento",
        "🖼️ Reconstrucción",
        "✨ Generación & Espacio Latente",
        "📏 Calidad de las Muestras",
        "📝 Conclusiones"
    ])
    vae = results["vae"]

    with tab_train:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MSE Reconstrucción (Dev)", f"{vae['dev_mse']:.5f}")
        c2.metric("MSE Reconstrucción (Test)", f"{vae['test_mse']:.5f}")
        c3.metric("Linear Probe F1 (Dev)", f"{vae['dev_probe_f1']:.4f}")
        c4.metric("Linear Probe F1 (Test)", f"{vae['test_probe_f1']:.4f}")
        with st.expander("📄 Elección y justificación del modelo: VAE vs. DCGAN"):
            nb(86)
        with st.expander("🏗️ Arquitectura y función objetivo"):
            nb(92, 94, 95)
        _, df_vae = training_logs()
        c_c1, c_c2, c_c3 = st.columns(3)
        for col, (title, tr, dv) in zip([c_c1, c_c2, c_c3], [("Pérdida total", "Train Total", "Dev Total"),
                                                              ("Reconstrucción (BCE)", "Train BCE", "Dev BCE"),
                                                              ("Divergencia KL", "Train KL", "Dev KL")]):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_vae["Época"], y=df_vae[tr], mode="lines+markers", name="Train", line=dict(color="#3b82f6")))
            fig.add_trace(go.Scatter(x=df_vae["Época"], y=df_vae[dv], mode="lines+markers", name="Dev", line=dict(color="#f59e0b")))
            fig.update_layout(title=title, xaxis_title="Época", height=320, margin=dict(l=0, r=0, t=35, b=10))
            col.plotly_chart(fig, width="stretch")
        nb(98)

    with tab_rec:
        nb(101, 102, 103, 104)

    with tab_lat:
        nb(105, 106, 107, 108, 109, 110, 111, 112, 113, 114)

    with tab_q:
        nb(115, 116, 117)

    with tab_conc:
        nb(118)

# ==============================================================================
# ETAPA 5: EVALUACIÓN FINAL & ÉTICA
# ==============================================================================
elif etapa == ETAPAS[4]:
    st.subheader("🏁 Etapa 5: Cierre, Comparación Integral, Evaluación en Test y Análisis Ético")
    tab_tab, tab_test, tab_lat, tab_eth, tab_conc = st.tabs([
        "🏆 Tabla Integral & Trade-off",
        "🧪 Evaluación en Test",
        "⚡ Latencia & Sobreajuste a Dev",
        "⚖️ Ética & Subgrupos",
        "📝 Conclusiones"
    ])
    champ_name = results["champion"]
    champ = MODELS[champ_name]
    df_models = pd.DataFrame(results["models"])

    with tab_tab:
        st.dataframe(df_models[["Modelo", "Etapa", "Arquitectura", "Parámetros", "Train Acc (%)", "Dev Acc (%)", "Dev Macro F1",
                                "Test Acc (%)", "Test Macro F1", "Brecha Dev-Test F1 (pp)", "Tiempo Entren. (s)"]]
                     .astype(str).replace({"nan": "—", "None": "—"}),
                     width="stretch", hide_index=True)
        st.plotly_chart(complexity_chart(df_models, "Complejidad vs. rendimiento en Dev (corrida del 21/11)"), width="stretch")
        nb(121)

    with tab_test:
        c1, c2, c3, c4 = st.columns(4)
        lo, hi = results["test_bootstrap_ci95"]
        c1.metric("Modelo Campeón", champ_name)
        c2.metric("Test Macro F1", f"{champ['Test Macro F1']:.4f}", delta=f"Dev: {champ['Dev Macro F1']:.4f}", delta_color="off")
        c3.metric("Test Accuracy", f"{champ['Test Acc (%)']:.2f}%")
        c4.metric("IC 95% Bootstrap (F1)", f"[{lo:.4f}, {hi:.4f}]")

        st.markdown("#### Matriz de Confusión del Campeón (Test Set, 2.700 imágenes)")
        tipo_cm = st.radio("Modalidad de Visualización:", ["Recall Normalizado (Tasa de Acierto)", "Conteos Absolutos (Muestras Reales)"], horizontal=True)
        st.plotly_chart(confusion_heatmap(results["test_confusion_matrix"], "Normalizado" in tipo_cm), width="stretch")

        st.markdown("#### Desempeño por Clase en Test")
        df_cls = pd.DataFrame(results["test_per_class"])
        c_t, c_g = st.columns([1, 1])
        c_t.dataframe(df_cls, width="stretch", hide_index=True)
        fig_cls = px.bar(df_cls.sort_values("Recall"), x="Recall", y="Clase", orientation="h", range_x=[0.8, 1],
                         text=df_cls.sort_values("Recall")["Recall"].map("{:.3f}".format), color="Recall", color_continuous_scale="Blues")
        fig_cls.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=10), coloraxis_showscale=False)
        c_g.plotly_chart(fig_cls, width="stretch")

    with tab_lat:
        lat = results["latency"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latencia por imagen (lote de 1)", f"{lat['ms_por_imagen_individual']:.2f} ms")
        c2.metric("Latencia por imagen (lotes de 64)", f"{lat['ms_por_imagen_lote64']:.3f} ms")
        c3.metric("Tamaño de los pesos (float32)", f"{lat['tamano_mb_float32']:.2f} MB")
        c4.metric("Dispositivo de medición", str(lat["dispositivo"]).upper())
        nb(132, 133, 134)

    with tab_eth:
        nb(135, 136, 137, 138)
        st.markdown("#### Umbrales de Revisión Humana (datos exportados)")
        st.dataframe(pd.DataFrame(results["human_review_thresholds"]), width="stretch", hide_index=True)

    with tab_conc:
        nb(139, 140, 141)

# ==============================================================================
# MLOPS LEDGER & HISTÓRICO DE EXPERIMENTOS
# ==============================================================================
elif etapa == ETAPAS[5]:
    st.subheader("📊 MLOps Experiment Ledger & Matriz de Trade-Off")
    st.markdown("Registro inmutable de corridas generado por `src/training/tracker.py`.")

    tab_runs, tab_tradeoff = st.tabs([
        "📋 Bitácora de Corridas (Ledger)",
        "⚖️ Matriz de Trade-off (F1 vs. Complejidad)"
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
            st.dataframe(df_l, width="stretch", hide_index=True, height=35 * (len(df_l) + 1) + 3)
            st.caption("EXP-014 a EXP-016 corresponden a la corrida oficial del 21/11; las anteriores son corridas exploratorias.")

            # ¿Por qué cambia el Train Acc del Baseline? (sale directo de los hiperparámetros registrados)
            base_runs = [e for e in ledger_data if e["model_name"].startswith("Baseline") and "max_iter" in e["hyperparameters"]]
            if base_runs:
                st.markdown("#### 🔎 ¿Qué explica las diferencias de Train Acc en el Baseline?")
                df_b = pd.DataFrame([{"Run": e["run_id"], "max_iter": e["hyperparameters"]["max_iter"],
                                      "C": e["hyperparameters"].get("C"), "Train Acc (%)": e["metrics"]["train_accuracy"],
                                      "Dev Acc (%)": e["metrics"]["dev_accuracy"]} for e in base_runs])
                alto = df_b[df_b["max_iter"] >= 300]
                bajo = df_b[df_b["max_iter"] < 300]
                c_txt, c_fig = st.columns([1, 1])
                with c_txt:
                    st.info(
                        f"En las {len(df_b)} corridas exploratorias del Baseline, lo que define el Train Acc es **`max_iter`** "
                        f"(cuántas iteraciones puede hacer el optimizador `lbfgs`):\n\n"
                        f"* **max_iter ≥ 300** ({len(alto)} corridas): el optimizador converge y el modelo **memoriza** Train → "
                        f"{alto['Train Acc (%)'].min():.1f}–{alto['Train Acc (%)'].max():.1f}%.\n"
                        f"* **max_iter ≤ 200** ({len(bajo)} corridas): se corta antes de converger, lo que actúa como "
                        f"regularización implícita → {bajo['Train Acc (%)'].min():.1f}–{bajo['Train Acc (%)'].max():.1f}%.\n\n"
                        f"`C` (entre {df_b['C'].min()} y {df_b['C'].max()}) casi no lo mueve, y el Dev Acc se mantiene en "
                        f"{df_b['Dev Acc (%)'].min():.1f}–{df_b['Dev Acc (%)'].max():.1f}% en todos los casos: más ajuste a Train "
                        f"no mejora la generalización (alta varianza).")
                with c_fig:
                    fig_mi = px.scatter(df_b, x="max_iter", y="Train Acc (%)", color="C", text="Run", hover_data=["Dev Acc (%)"],
                                        color_continuous_scale="Viridis")
                    fig_mi.update_traces(marker=dict(size=13), textposition="middle right", textfont=dict(size=10))
                    fig_mi.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=10), xaxis=dict(range=[50, 600]))
                    st.plotly_chart(fig_mi, width="stretch")

            st.markdown("#### 🧾 Detalle por experimento: hiperparámetros y qué cambió")
            st.caption("Cada fila se compara con la corrida anterior del mismo modelo. Los datos salen de `experiments_ledger.json`.")

            def family(name):
                return "Baseline" if name.startswith("Baseline") else name.split("(")[0].strip()

            for i, e in enumerate(ledger_data):
                prev = next((p for p in ledger_data[i + 1:] if family(p["model_name"]) == family(e["model_name"])), None)
                m = e["metrics"]
                label = (f"{e['run_id']} · {e['timestamp'][:10]} · {e['model_name']} — Train {m.get('train_accuracy', 0):.2f}% · "
                         f"Dev {m.get('dev_accuracy', 0):.2f}% · F1 {m.get('macro_f1', 0):.4f}")
                with st.expander(label):
                    hp, hp_prev = e.get("hyperparameters", {}), (prev or {}).get("hyperparameters", {})
                    ds, ds_prev = e.get("dataset", {}), (prev or {}).get("dataset", {})
                    rows = [{"Parámetro": k, "Valor": str(hp.get(k, "—")),
                             f"Anterior ({prev['run_id']})" if prev else "Anterior": str(hp_prev.get(k, "—")) if prev else "—"}
                            for k in list(hp) + [k for k in hp_prev if k not in hp]]
                    rows += [{"Parámetro": f"dataset.{k}", "Valor": str(ds.get(k, "—")),
                              f"Anterior ({prev['run_id']})" if prev else "Anterior": str(ds_prev.get(k, "—")) if prev else "—"}
                             for k in list(ds) + [k for k in ds_prev if k not in ds]]
                    c_hp, c_ch = st.columns([1, 1])
                    c_hp.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                    with c_ch:
                        if prev is None:
                            st.markdown("**Primera corrida registrada de este modelo.**")
                        else:
                            cambios = [r for r in rows if r["Valor"] != list(r.values())[2]]
                            pm = prev["metrics"]
                            st.markdown(f"**Cambios respecto de {prev['run_id']}:**")
                            if cambios:
                                st.markdown("\n".join(f"* `{r['Parámetro']}`: {list(r.values())[2]} → **{r['Valor']}**" for r in cambios))
                            else:
                                st.markdown("* El ledger no registra cambios de hiperparámetros ni de datos; la diferencia "
                                            "viene de algo no registrado (código, semilla o cantidad de épocas).")
                            st.markdown(
                                f"**Efecto:** Train Acc {pm.get('train_accuracy', 0):.2f}% → **{m.get('train_accuracy', 0):.2f}%** · "
                                f"Dev Acc {pm.get('dev_accuracy', 0):.2f}% → **{m.get('dev_accuracy', 0):.2f}%** · "
                                f"Gap {pm.get('gap', 0):.2f} → **{m.get('gap', 0):.2f}** pp")
                        st.markdown(f"**Diagnóstico:** {e.get('diagnosis', '—')}")
                        if e.get("notes"):
                            st.markdown(f"**Notas:** {e['notes']}")
        else:
            st.info("Aún no hay registros en `experiments_ledger.json`.")

    with tab_tradeoff:
        df_models = pd.DataFrame(results["models"]).dropna(subset=["Train Acc (%)"])
        st.plotly_chart(complexity_chart(df_models, "Trade-off: Macro F1 (Dev) vs. Parámetros (tamaño del punto = tiempo de "
                                         "entrenamiento)", size_by_time=True), width="stretch")
        fig_g = px.bar(
            df_models, x="Modelo", y="Brecha Train-Dev (pp)",
            title="Evolución de la Brecha de Varianza (Train - Dev, pp)",
            color="Brecha Train-Dev (pp)", color_continuous_scale="Reds"
        )
        fig_g.update_layout(height=480)
        st.plotly_chart(fig_g, width="stretch")

# ==============================================================================
# TP ENTREGADO: NOTEBOOK COMPLETO, UNA PESTAÑA POR ETAPA
# ==============================================================================
elif etapa == ETAPAS[6]:
    st.subheader("📓 Trabajo Práctico Entregado — Notebook Ejecutado en Google Colab (21/11/2026)")
    show_code = st.toggle("Mostrar código de las celdas", value=False)
    stages, current = [], []
    for i, cell in enumerate(NB_CELLS):
        if cell["cell_type"] == "markdown" and re.search(r"^# Etapa \d", "".join(cell["source"]), re.M) and current:
            stages.append(current)
            current = []
        current.append(i)
    stages.append(current)
    for tab, idxs in zip(st.tabs([f"Etapa {k}" for k in range(1, len(stages) + 1)]), stages):
        with tab:
            nb(*idxs, show_code=show_code)
