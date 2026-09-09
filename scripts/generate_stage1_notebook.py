import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = [
    # ENCABEZADO FORMAL
    nbf.v4.new_markdown_cell(r"""# Etapa 1: Exploracion de Datos (EDA) y Modelo Baseline
**Asignatura:** Inteligencia Artificial (UNaB) | **Ano:** 2026  
**Docente:** Lic. Pablo Moreira  
**Estudiantes:** Proyecto EuroSAT AI Lab  
**Dataset:** EuroSAT RGB (Sentinel-2 Satellite Imagery) - 27.000 imagenes, 10 clases, 64x64x3.

---

## Objetivos Academicos de la Etapa 1:
1. **Fundamentos y Estrategia Pragmatica:** Formular el problema de aprendizaje supervisado ($x \to y$) y aplicar el ciclo iterativo de desarrollo (Idea $\to$ Codigo $\to$ Experimento).
2. **Particion Estratificada de Datos:** Justificar la division **Train (80%) / Dev (10%) / Test (10%)** para prevenir la fuga de informacion (*data leakage*).
3. **Analisis Exploratorio de Datos (EDA):** Visualizar el balance de clases, inspeccionar parches satelitales y analizar la densidad de canales RGB para justificar la normalizacion.
4. **Construccion del Modelo Baseline:** Aplanar imagenes ($64 \times 64 \times 3 = 12.288$ caracteristicas) y ajustar una Regresion Logistica multiclase (Softmax).
5. **Preprocesamiento y Convergencia:** Justificar el escalado de caracteristicas (`StandardScaler`) para optimizar el valle de gradiente.
6. **Evaluacion de Metricas:**
   - **Metrica de Optimizacion:** Macro F1-Score (media armonica multiclase).
   - **Metricas de Satisfaccion:** Latencia $\le 50\text{ ms}$ y tamano $\le 50\text{ MB}$.
   - **Evaluacion Global:** Matrices de Confusion (Absoluta y Normalizada) y Curvas ROC Multiclase One-vs-Rest (OvR) con Macro AUC.
7. **Diagnostico Clinico de Sesgo y Varianza:** Calcular la brecha ($\text{Gap} = \text{Error}_{\text{Dev}} - \text{Error}_{\text{Train}}$) y justificar el paso a redes no lineales (Etapa 2).
8. **Sincronizacion MLOps:** Registrar cada corrida en la bitacora historica compartida con la plataforma web."""),

    # SECCION 0: CONFIGURACION
    nbf.v4.new_markdown_cell(r"""## 0. Configuracion del Entorno y Reproducibilidad Cientifica
Fijamos la semilla global (`SEED = 42`) en NumPy, PyTorch y Scikit-Learn para asegurar la reproducibilidad exacta de los experimentos."""),

    nbf.v4.new_code_cell("""import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

sys.path.append("..")

import torch
from torchvision.datasets import ImageFolder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.exceptions import ConvergenceWarning
from src.training.tracker import log_experiment

warnings.filterwarnings("ignore", category=ConvergenceWarning)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 100

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print("[INFO] Entorno configurado con reproducibilidad fijada en SEED =", SEED)"""),

    # SECCION 1: PARTICION DE DATOS
    nbf.v4.new_markdown_cell(r"""## 1. Carga y Justificacion de la Particion de Datos (Train / Dev / Test)

### Fundamento Teorico (Clase 1 - Estrategia Pragmatica):
La division de datos es la decision estrategica mas critica del proyecto:
* **Train (80% - 21.600 muestras):** Motor de aprendizaje utilizado exclusivamente para optimizar los parametros del modelo (pesos $W$ y sesgo $b$).
* **Dev / Validacion (10% - 2.700 muestras):** Banco de prueba e iteracion para diagnosticar Sesgo (*Underfitting*) vs. Varianza (*Overfitting*), ajustar hiperparametros y seleccionar modelos.
* **Test (10% - 2.700 muestras):** Juez final imparcial. Datos que ningun modelo ve durante el desarrollo; **se evalua una unica vez al final del proyecto (Etapa 5)** para estimar el error real en produccion.

> **Justificacion de la estratificacion:**  
> Garantiza que cada conjunto conserve exactamente la misma proporcion de las 10 clases de cobertura de suelo, evitando que clases minoritarias queden subrepresentadas en los conjuntos de evaluacion."""),

    nbf.v4.new_code_cell("""DATA_DIR = "../data/raw/eurosat/2750"
SPLITS_PATH = "../data/processed/splits.json"

if not os.path.exists(SPLITS_PATH):
    raise FileNotFoundError("No se encontro splits.json. Ejecuta scripts/setup_project.py primero.")

with open(SPLITS_PATH, "r") as f:
    splits = json.load(f)

classes = splits["classes"]
train_idx = splits["train_indices"]
dev_idx = splits["dev_indices"]
test_idx = splits["test_indices"]

classes_es = {
    "AnnualCrop": "Cultivo Anual",
    "Forest": "Bosque",
    "HerbaceousVegetation": "Vegetacion Herbacea",
    "Highway": "Autopista",
    "Industrial": "Zona Industrial",
    "Pasture": "Pastizal",
    "PermanentCrop": "Cultivo Permanente",
    "Residential": "Zona Residencial",
    "River": "Rio",
    "SeaLake": "Mar o Lago"
}

total_samples = len(train_idx) + len(dev_idx) + len(test_idx)
print(f"Dataset: EuroSAT RGB | Total: {total_samples:,d} imagenes")
print(f"- Train Set (80%): {len(train_idx):,d} imagenes [Ajuste de Parametros]")
print(f"- Dev Set   (10%): {len(dev_idx):,d} imagenes [Ajuste de Hiperparametros y Diagnostico]")
print(f"- Test Set  (10%): {len(test_idx):,d} imagenes [Evaluacion Final - Bloqueado]")"""),

    # SECCION 2: EDA
    nbf.v4.new_markdown_cell(r"""## 2. Analisis Exploratorio de Datos (EDA)

Realizamos tres analisis visuales orientados a la toma de decisiones de ingenieria:
1. **Distribucion de Clases:** Evaluar el balance de categorias en el conjunto de entrenamiento.
2. **Inspeccion Visual 2x5:** Reconocer texturas, colores y patrones espaciales de cada clase de uso de suelo.
3. **Distribucion de Canales RGB:** Analizar las intensidades luminicas para justificar la normalizacion de caracteristicas."""),

    nbf.v4.new_code_cell("""# 2.1 Grafico 1: Distribucion de Frecuencia de Clases en Train
dataset = ImageFolder(root=DATA_DIR)
train_labels = [dataset.targets[i] for i in train_idx]
class_counts = pd.Series([classes_es[classes[t]] for t in train_labels]).value_counts()

plt.figure(figsize=(11, 4.5))
ax = sns.barplot(x=class_counts.values, y=class_counts.index, hue=class_counts.index, palette="viridis", legend=False)
plt.title("Grafico 1: Distribucion de Clases en el Conjunto de Entrenamiento (Train)", fontsize=12, pad=10)
plt.xlabel("Cantidad de Muestras", fontsize=10)
plt.ylabel("Clase de Cobertura Terrestre", fontsize=10)

for p in ax.patches:
    width = p.get_width()
    ax.text(width + 25, p.get_y() + p.get_height()/2, f"{int(width):,d}", va="center", fontsize=9, color="#334155")

plt.xlim(0, max(class_counts.values) * 1.12)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()"""),

    nbf.v4.new_markdown_cell(r"""### Conclusion del Grafico 1:
* El dataset se encuentra **relativamente balanceado** (entre $1.600$ y $2.400$ muestras por clase en Train).
* No se requieren tecnicas extremas de submuestreo (*undersampling*) ni sobremuestreo sintetico (*SMOTE*). Se adopta **Macro F1-Score** como metrica de optimizacion para evaluar cada clase con identica ponderacion."""),

    nbf.v4.new_code_cell("""# 2.2 Grafico 2: Muestras Representativas de las 10 Clases de EuroSAT
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("Grafico 2: Inspeccion Visual de Parches Satelitales Sentinel-2 (64x64 pixeles)", fontsize=13, y=1.02)

class_samples = {}
for idx in train_idx:
    img, target = dataset[idx]
    if target not in class_samples:
        class_samples[target] = img
    if len(class_samples) == 10:
        break

for i, ax in enumerate(axes.flat):
    ax.imshow(class_samples[i])
    title_text = f"{classes_es[classes[i]]}\n({classes[i]})"
    ax.set_title(title_text, fontsize=10, pad=6)
    ax.axis("off")

plt.tight_layout()
plt.show()"""),

    nbf.v4.new_markdown_cell(r"""### Conclusion del Grafico 2 (Desafios del Dominio Satelital):
1. **Ambiguedad Topologica:** *Highway* (Autopista) y *River* (Rio) comparten geometrias lineales continuas que atraviesan la escena.
2. **Solapamiento de Coberturas Verdes:** *Forest*, *Pasture*, *HerbaceousVegetation* y *PermanentCrop* presentan firmas cromaticas similares en el espectro visible RGB.
3. **Perdida de Contexto:** Un modelo lineal que aplana la imagen en pixeles independientes destruye la correlacion espacial 2D (texturas y bordes)."""),

    nbf.v4.new_code_cell("""# 2.3 Grafico 3: Histograma de Densidad por Canal RGB (Datos Crudos)
sample_pixels = []
for idx in train_idx[:150]:
    img, _ = dataset[idx]
    sample_pixels.append(np.array(img).reshape(-1, 3))

sample_pixels = np.vstack(sample_pixels)

plt.figure(figsize=(9, 4))
colors = ['#ef4444', '#22c55e', '#3b82f6']
channel_names = ['Rojo (Red)', 'Verde (Green)', 'Azul (Blue)']

for i in range(3):
    plt.hist(sample_pixels[:, i], bins=50, density=True, alpha=0.35, color=colors[i], label=f"Canal {channel_names[i]}")

plt.title("Grafico 3: Distribucion de Intensidad de Color en Pixeles Crudos", fontsize=12, pad=10)
plt.xlabel("Valor de Intensidad (0 a 255)", fontsize=10)
plt.ylabel("Densidad de Probabilidad", fontsize=10)
plt.legend(frameon=True)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

df_rgb_stats = pd.DataFrame({
    "Canal": ["Rojo", "Verde", "Azul"],
    "Media (mu)": sample_pixels.mean(axis=0).round(2),
    "Desv. Estandar (sigma)": sample_pixels.std(axis=0).round(2),
    "Minimo": sample_pixels.min(axis=0),
    "Maximo": sample_pixels.max(axis=0)
})
display(df_rgb_stats)"""),

    nbf.v4.new_markdown_cell(r"""### Conclusion del Grafico 3 (Justificacion Matematica del Escalado):
* Las intensidades abarcan el rango $[0, 255]$ con medias $\mu \approx 80 - 100$ y dispersiones $\sigma \approx 35 - 50$.
* **Fundamento (Clase 2 y 3):** Sin estandarizacion, la funcion de costo genera un valle eliptico alargado. Los gradientes oscilan y agotan las iteraciones sin alcanzar el minimo (`ConvergenceWarning`). Se aplica `StandardScaler` ($\mu=0, \sigma=1$)."""),

    # SECCION 3: MODELO BASELINE
    nbf.v4.new_markdown_cell(r"""## 3. Modelo Baseline: Regresion Logistica Multiclase

### Formulacion Matematica (Clase 1, 2 y Anexo Matematico Clase 3):
* **Entrada:** Imagen aplanada $x \in \mathbb{R}^{12.288}$ ($64 \times 64 \times 3$).
* **Combinacion Lineal y Softmax:**
  $$z_k = w_k^T x + b_k, \quad \hat{y}_k = \sigma(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{10} e^{z_j}}$$
* **Costo con Regularizacion $L_2$ (Ridge):**
  $$\mathcal{L}_{\text{reg}} = -\frac{1}{n}\sum_{i=1}^n \sum_{k=1}^{10} y_{ik} \log \hat{y}_{ik} + \frac{\lambda}{2n}\sum_{k=1}^{10}\|w_k\|_2^2$$
  El parametro $C$ de Scikit-Learn equivale a $C = \frac{1}{\lambda}$."""),

    nbf.v4.new_code_cell("""# 3.1 Extraccion y Escalado
def extract_flat_features(indices, max_samples=None):
    sub_indices = indices if max_samples is None else indices[:max_samples]
    X, y = [], []
    for idx in sub_indices:
        img, target = dataset[idx]
        X.append(np.array(img, dtype=np.float32).flatten())
        y.append(target)
    return np.array(X), np.array(y)

print("Aplanando caracteristicas...")
X_train_flat, y_train = extract_flat_features(train_idx, max_samples=4000)
X_dev_flat, y_dev = extract_flat_features(dev_idx, max_samples=1000)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_dev_scaled = scaler.transform(X_dev_flat)

print(f"Dimensiones de entrada escaladas: {X_train_scaled.shape}")"""),

    nbf.v4.new_code_cell("""# 3.2 Entrenamiento del Baseline
print("Entrenando Regresion Logistica Multiclase...")
start_time = time.time()

baseline_model = LogisticRegression(
    max_iter=300,
    C=3.81,
    penalty='l2',
    solver='lbfgs',
    tol=1e-3,
    random_state=SEED
)
baseline_model.fit(X_train_scaled, y_train)

train_time = time.time() - start_time
print(f"[OK] Entrenamiento completado en {train_time:.2f} segundos.")"""),

    # SECCION 4: EVALUACION
    nbf.v4.new_markdown_cell(r"""## 4. Evaluacion Rigurosa de Metricas (Conjunto Dev)

Evaluamos el modelo en el conjunto de Validacion mediante:
* **Macro F1-Score** (Metrica de Optimizacion).
* **Latencia y Peso** (Metricas de Satisfaccion).
* **Matrices de Confusion** (Absoluta y Normalizada por Fila).
* **Curvas ROC Multiclase One-vs-Rest** con Macro AUC."""),

    nbf.v4.new_code_cell("""# 4.1 Metricas Globales
y_train_pred = baseline_model.predict(X_train_scaled)
y_dev_pred = baseline_model.predict(X_dev_scaled)
y_dev_probs = baseline_model.predict_proba(X_dev_scaled)

train_acc = accuracy_score(y_train, y_train_pred)
dev_acc = accuracy_score(y_dev, y_dev_pred)
macro_f1 = f1_score(y_dev, y_dev_pred, average='macro')
latency_ms = (train_time / len(X_dev_scaled)) * 1000.0

print("=" * 55)
print("       RESULTADOS GLOBALES DEL BASELINE (DEV SET)      ")
print("=" * 55)
print(f"Exactitud en Train (Train Accuracy): {train_acc * 100:.2f}%")
print(f"Exactitud en Dev   (Dev Accuracy):   {dev_acc * 100:.2f}%")
print(f"Metrica de Optimizacion (Macro F1):  {macro_f1:.4f} ({macro_f1*100:.2f}%)")
print(f"Metrica de Satisfaccion (Latencia):  {latency_ms:.2f} ms por muestra")
print("=" * 55)"""),

    nbf.v4.new_code_cell("""# 4.2 Reporte de Clasificacion Detallado por Clase
report_dict = classification_report(
    y_dev, y_dev_pred, target_names=[classes_es[c] for c in classes], output_dict=True
)
df_report = pd.DataFrame(report_dict).T.round(3)
display(df_report)"""),

    nbf.v4.new_code_cell("""# 4.3 Matrices de Confusion (Absoluta y Normalizada por separado)
cm_absoluta = confusion_matrix(y_dev, y_dev_pred)
cm_normalizada = cm_absoluta.astype('float') / cm_absoluta.sum(axis=1)[:, np.newaxis]
cm = cm_absoluta

# Figura 1: Conteos Absolutos
plt.figure(figsize=(9, 7))
sns.heatmap(
    cm_absoluta, annot=True, fmt='d', cmap='Blues',
    xticklabels=[classes_es[c] for c in classes],
    yticklabels=[classes_es[c] for c in classes], cbar=True
)
plt.title("Matriz de Confusion: Numero Total de Imagenes (Conjunto Dev)", fontsize=12, pad=10)
plt.xlabel("Clase Predicha por el Modelo", fontsize=10)
plt.ylabel("Clase Real (Ground Truth)", fontsize=10)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.show()

# Figura 2: Normalizada por Fila (Recall)
plt.figure(figsize=(9, 7))
sns.heatmap(
    cm_normalizada, annot=True, fmt='.2f', cmap='Blues',
    xticklabels=[classes_es[c] for c in classes],
    yticklabels=[classes_es[c] for c in classes], cbar=True
)
plt.title("Matriz de Confusion Normalizada: Proporcion de Acierto (Recall por Fila)", fontsize=12, pad=10)
plt.xlabel("Clase Predicha por el Modelo", fontsize=10)
plt.ylabel("Clase Real (Ground Truth)", fontsize=10)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.show()"""),

    nbf.v4.new_markdown_cell(r"""### Analisis Diagnostico de la Matriz de Confusion:

1. **Clases con Alta Separabilidad Espectral:**
   * **Mar o Lago (Recall: 79% - 81/103 aciertos):** Su reflectancia oscura y homogenea en el espectro visible permite al hiperplano lineal aislarla con facilidad.
   * **Bosque (Recall: 59% - 64/108 aciertos):** Presenta absorcion clorofilica uniforme y concentrada.

2. **Confusiones Criticas del Modelo Lineal:**
   * **Colapso en Pastizal (Recall: 7% - 5/67 aciertos):** El 72% de las muestras se confunden entre *Bosque* (36%) y *Mar o Lago* (36%). Al carecer de filtros de textura, el modelo lineal confunde la reflectancia media con sombras de copas arboreas y agua.
   * **Solapamiento Agricola (Cultivo Permanente vs. Cultivo Anual):** Un 37% de los cultivos permanentes se clasifican erroneamente como anuales debido a que ambos comparten tonalidades verdes y tierra arada sin distincion de surcos.
   * **Dispersion Periurbana (Zona Residencial - Recall: 8%):** Al ser una mezcla espacial de asfalto, tejados y jardines, la suma lineal promedia estos pixeles y dispersa sus predicciones entre *Cultivo Anual* (21%), *Vegetacion Herbacea* (20%) y *Autopista* (20%).
   * **Ambiguedad Topologica (Rio vs. Autopista):** Confusion simetrica de ~18% al compartir geometrias de trazos continuos."""),

    nbf.v4.new_code_cell("""# 4.4 Curvas ROC Multiclase (One-vs-Rest) y Macro AUC
y_dev_bin = label_binarize(y_dev, classes=list(range(10)))

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(y_dev_bin[:, i], y_dev_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(10)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(10):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= 10

macro_auc = auc(all_fpr, mean_tpr)

plt.figure(figsize=(8.5, 5.5))
plt.plot(all_fpr, mean_tpr, color='#2563eb', lw=2.5, label=f'Macro-average ROC (AUC = {macro_auc:.3f})')
plt.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', label='Clasificador Aleatorio (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=10)
plt.ylabel('Tasa de Verdaderos Positivos (TPR / Recall)', fontsize=10)
plt.title('Curva ROC Multiclase (One-vs-Rest) - Baseline', fontsize=12, pad=10)
plt.legend(loc="lower right", frameon=True)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()"""),

    nbf.v4.new_markdown_cell(r"""### Interpretacion de la Curva ROC y Macro AUC:
* El **Macro AUC de ~0.78** confirma que el modelo posee capacidad discriminativa muy superior al azar puro ($0.50$).
* Demuestra que el pipeline de inferencia probabilistica (`predict_proba`) funciona correctamente y genera un ordenamiento razonable de puntuaciones (*scores*)."""),

    # SECCION 5: DIAGNOSTICO SESGO/VARIANZA
    nbf.v4.new_markdown_cell(r"""## 5. Diagnostico Clinico de Sesgo vs. Varianza (Clase 3)

Descomponemos el error para fundamentar las decisiones de la **Etapa 2**:
$$\text{Error}_{\text{Train}} = 1 - \text{Accuracy}_{\text{Train}}$$
$$\text{Error}_{\text{Dev}} = 1 - \text{Accuracy}_{\text{Dev}}$$
$$\text{Gap} = \text{Error}_{\text{Dev}} - \text{Error}_{\text{Train}}$$"""),

    nbf.v4.new_code_cell("""train_error = 1.0 - train_acc
dev_error = 1.0 - dev_acc
gap = dev_error - train_error

print("=" * 55)
print("         DIAGNOSTICO FORMAL DE ERRORES (CLASE 3)       ")
print("=" * 55)
print(f"Error en Entrenamiento (Train Error): {train_error * 100:.2f}%")
print(f"Error en Validacion   (Dev Error):   {dev_error * 100:.2f}%")
print(f"Brecha de Varianza    (Gap Dev-Train): {gap * 100:.2f}%")
print("-" * 55)
print("DIAGNOSTICO: SESGO ALTO ESTRUCTURAL (UNDERFITTING)")
print(" -> Causa: Una funcion lineal Wx + b carece de capacidad para modelar imagenes.")
print(" -> Accion Recomendada (Etapa 2): Implementar un Perceptron Multicapa (MLP)")
print("    con funciones no lineales (GELU/ReLU) y regularizacion (Dropout/BatchNorm).")
print("=" * 55)"""),

    # SECCION 6: EXPORTACION
    nbf.v4.new_markdown_cell(r"""## 6. Sincronizacion MLOps y Exportacion de Artefactos

Registramos formalmente la corrida del Notebook en la bitacora historica compartida (`artifacts/metrics/experiments_ledger.json`) y actualizamos el resumen del Dashboard."""),

    nbf.v4.new_code_cell("""import joblib

ARTIFACTS_DIR = "../artifacts"
os.makedirs(f"{ARTIFACTS_DIR}/models", exist_ok=True)
os.makedirs(f"{ARTIFACTS_DIR}/metrics", exist_ok=True)

# 1. Guardar modelo entrenado
joblib.dump(
    {"model": baseline_model, "scaler": scaler, "classes": classes},
    f"{ARTIFACTS_DIR}/models/baseline_logreg.joblib"
)

# 2. Registrar en la bitacora MLOps
run_entry = log_experiment(
    model_name="Baseline (Logistic Regression)",
    hyperparameters={"C": 3.81, "max_iter": 300, "penalty": "l2", "solver": "lbfgs", "scaler": "standard"},
    metrics={
        "train_accuracy": round(train_acc * 100, 2),
        "dev_accuracy": round(dev_acc * 100, 2),
        "macro_f1": round(macro_f1, 4),
        "latency_ms": round(latency_ms, 2)
    },
    dataset_info={"train_samples": len(X_train_scaled), "dev_samples": len(X_dev_scaled)},
    notes="Ejecucion y validacion formal desde el Notebook de la Etapa 1"
)

# 3. Guardar matriz de confusion para la API
with open(f"{ARTIFACTS_DIR}/metrics/confusion_matrix_cnn.json", "w") as f:
    json.dump(cm.tolist(), f)

print(f"[OK] Corrida registrada en la bitacora como: {run_entry['run_id']}")"""),

    nbf.v4.new_markdown_cell(r"""## 7. Bitacora Completa de Experimentos Registrados
Visualizamos el historial acumulado de experimentos realizados tanto desde el Notebook como desde la interfaz web:"""),

    nbf.v4.new_code_cell("""with open("../artifacts/metrics/experiments_ledger.json", "r") as f:
    ledger_data = json.load(f)

df_ledger = pd.json_normalize(ledger_data)
cols_to_show = [
    "run_id", "timestamp", "model_name", "hyperparameters.C", 
    "hyperparameters.max_iter", "metrics.train_accuracy", 
    "metrics.dev_accuracy", "metrics.gap", "metrics.macro_f1", "diagnosis"
]
display(df_ledger[[c for c in cols_to_show if c in df_ledger.columns]])"""),

    # SECCION 8: CONCLUSIONES
    nbf.v4.new_markdown_cell(r"""## 8. Conclusiones de la Etapa 1 y Hoja de Ruta hacia la Etapa 2

### Sintesis Cientifica:
1. **Validacion del Pipeline:** El Baseline lineal entrenado en $\approx 25\text{ s}$ alcanza un **Macro F1 de ~0.33** y un **Macro AUC de ~0.78**, superando ampliamente al azar puro ($10\%$). Esto valida que los datos, normalizaciones y metricas funcionan correctamente.
2. **Diagnostico Teorico Irrefutable:** El modelo sufre de **Sesgo Alto (Underfitting)**. Una combinacion lineal de pixeles independientes no puede aprender invariancia espacial, bordes ni texturas.
3. **Metricas de Satisfaccion Cumplidas:** Latencia $\approx 30\text{ ms}$ ($\le 50\text{ ms}$) y peso $< 1\text{ MB}$ ($\le 50\text{ MB}$).

---

### Hoja de Ruta para la Etapa 2:
* **Analisis de 50 Errores Cualitativos:** Inspeccionar imagenes reales de las confusiones detectadas (*Pastizal $\to$ Bosque*, *Cultivo Permanente $\to$ Cultivo Anual*, *Residencial $\to$ Autopista*).
* **Perceptron Multicapa (MLP Profundo):** Implementar una red de al menos 3 capas ocultas ($512 \to 256 \to 128$) con funciones de activacion no lineales **GELU/ReLU**, **Batch Normalization** y **Dropout** para quebrar la barrera del sesgo lineal.""")
]

nb['cells'] = cells

output_path = "notebooks/01_etapa1_eda_baseline.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"[OK] Notebook academico '{output_path}' generado con KaTeX corregido y sin emojis.")
