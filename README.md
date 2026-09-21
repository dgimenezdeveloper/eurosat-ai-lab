# 🛰️ EuroSAT AI Lab — Land Cover Multi-Class Satellite Vision & MLOps Platform

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgimenezdeveloper/eurosat-ai-lab/blob/main/notebooks/TP_IA2026_BARRERAS_PORCHIA_PAAL_GIMENEZ_GRUPO-5.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Deep_Learning-PyTorch_2.2-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Machine_Learning-Scikit--Learn-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit_App-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-EuroSAT_RGB_(27K)-2BAE66.svg)](https://github.com/phelber/eurosat)

> End-to-end computer vision laboratory and interactive MLOps platform for Land Use and Land Cover (LULC) multi-class classification on European Space Agency (ESA) Sentinel-2 satellite imagery. Features a unified ecosystem with a PyTorch/Scikit-Learn modeling pipeline, a FastAPI inference engine, an interactive React 19 tuning dashboard, and a Streamlit scientific explorer.
>
> 🌐 **Quick Navigation / Navegación Rápida:** [English Documentation](#-english-documentation) | [Documentación en Español](#-documentación-en-español)

---

## 🌐 English Documentation

### 1. Executive Summary & Problem Formulation
Accurate monitoring of surface land use and vegetation cover is critical for agricultural forecasting, urban planning, and environmental conservation. 

**EuroSAT AI Lab** addresses this multi-class supervised computer vision problem ($X \to Y$) using optical satellite imagery from the ESA Sentinel-2 mission:
- **Input Space ($X$):** Optical satellite patches with dimensions of $64 \times 64$ pixels and 3 visible color bands (RGB), flattened into vectors of $12,288$ features or treated as $64 \times 64 \times 3$ tensors.
- **Output Space ($Y$):** 10 mutually exclusive land use and land cover classes:
  1. *AnnualCrop* (Annual Crop)
  2. *Forest* (Forest)
  3. *HerbaceousVegetation* (Herbaceous Vegetation)
  4. *Highway* (Highway / Road)
  5. *Industrial* (Industrial Area)
  6. *Pasture* (Pasture)
  7. *PermanentCrop* (Permanent Crop)
  8. *Residential* (Residential Area)
  9. *River* (River)
  10. *SeaLake* (Sea or Lake)

#### Computer Vision Domain Challenges
- **Spectral Ambiguity:** Vegetative categories (*Forest*, *Pasture*, *HerbaceousVegetation*, *PermanentCrop*) share overlapping green reflectance signatures in the visible spectrum.
- **Topological Confusion:** Linear man-made infrastructure (*Highway*) and natural waterways (*River*) share continuous edge geometries requiring spatial texture context.
- **Structural Bias in Linear Models:** Flattening $64 \times 64 \times 3$ images into isolated pixels destroys 2D spatial correlations, making linear models incapable of detecting complex textures.

---

### 2. Evaluation System & MLOps Governance

#### A. Deterministic Stratified Split (80 / 10 / 10)
To avoid data leakage and preserve class distribution across sets, the 27,000 images are split deterministically (`seed = 42`):
- **Train Set (80% — 21,600 images):** Reserved strictly for model parameter optimization ($W, b$).
- **Dev / Validation Set (10% — 2,700 images):** Used for architecture iteration, hyperparameter tuning (regularization, learning rates), and bias/variance error diagnosis.
- **Test Set (10% — 2,700 images):** Isolated and locked until the final project stage (Stage 5) to ensure an unbiased estimate of generalization error.

#### B. Single-Number Optimization & Satisficing Metrics
- **Primary Optimization Metric:** **Macro F1-Score** on the Dev set, ensuring equal weighting across all 10 land cover classes regardless of minor sample size variations.
- **Operational Constraints (Satisficing Metrics):**
  - **Inference Latency:** $\le 50\text{ ms}$ per sample on a standard CPU.
  - **Serialized Model Artifact:** $\le 100\text{ MB}$ on disk.
- **Diagnostic Metrics:** Normalized Confusion Matrices (Row Recall) and Multiclass One-vs-Rest (OvR) ROC Curves with Macro-average AUC.

#### C. Invariable MLOps Experiment Ledger
Every experiment run—whether executed via JupyterLab or triggered from the React frontend—is logged into `artifacts/metrics/experiments_ledger.json`, capturing:
- Unique run ID (e.g., `EXP-001`), timestamp, and model architecture.
- Full hyperparameter configuration (Regularization $C$, Solver, Penalty $L_1/L_2$, Scaler type).
- Train/Dev accuracy, generalization gap, Macro F1, and inference latency.
- Automated clinical diagnosis: *Structural Underfitting* vs. *High Variance Overfitting*.

---

### 3. Architecture & Tech Stack

```
┌────────────────────────────────────────────────────────────────────────┐
│                        EUROSAT AI LAB TOPOLOGY                         │
├───────────────────┬────────────────────────────────────────────────────┤
│ Modeling & Data   │ Python 3.10+, PyTorch 2.2, torchvision,            │
│                   │ Scikit-Learn 1.3, NumPy, Pandas, Pillow, Joblib    │
├───────────────────┼────────────────────────────────────────────────────┤
│ API & Telemetry   │ FastAPI 0.110, Uvicorn, Pydantic v2, CORS          │
├───────────────────┼────────────────────────────────────────────────────┤
│ User Interfaces   │ Frontend: React 19, Vite 8, Tailwind CSS v4,       │
│                   │ Base UI primitives, Lucide Icons (Port 5173)       │
│                   │ Science Console: Streamlit 1.32 (Port 8501)        │
├───────────────────┼────────────────────────────────────────────────────┤
│ Research Notebooks│ JupyterLab, Google Colab Integration (Port 8888)   │
├───────────────────┼────────────────────────────────────────────────────┤
│ Environment       │ VS Code DevContainers, Docker (PyTorch CUDA base)  │
└───────────────────┴────────────────────────────────────────────────────┘
```

---

### 4. Machine Learning 5-Stage Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROGRESSIVE EXPERIMENTAL ROADMAP                         │
├───────────────────┬─────────────────────────────────┬───────────────────────┤
│ Stage             │ Architecture & Methods          │ Scientific Purpose    │
├───────────────────┼─────────────────────────────────┼───────────────────────┤
│ 1. Baseline       │ Multiclass Logistic Regression  │ Linear reference floor│
│    (Completed)    │ (Softmax, StandardScaler, L-BFGS)│ and pipeline check.   │
├───────────────────┼─────────────────────────────────┼───────────────────────┤
│ 2. Deep MLP       │ Dense Neural Network            │ Overcome linear bias  │
│    (Next)         │ (3+ layers, GELU/ReLU, Dropout) │ with non-linear units.│
├───────────────────┼─────────────────────────────────┼───────────────────────┤
│ 3. 2D CNN         │ Convolutional Neural Network    │ Extract hierarchical  │
│                   │ (Conv2D, MaxPool, Augmentation) │ textures and edges.   │
├───────────────────┼─────────────────────────────────┼───────────────────────┤
│ 4. Latent VAE     │ Variational Autoencoder         │ Model representation  │
│    (Exploratory)  │ (Probabilistic Encoder/Decoder) │ space & generation.   │
├───────────────────┼─────────────────────────────────┼───────────────────────┤
│ 5. Test Synthesis │ Final Evaluation on Test Set    │ Trade-off analysis    │
│    (Closure)      │ (Single-run benchmark)          │ and final report.     │
└───────────────────┴─────────────────────────────────┴───────────────────────┘
```

#### Stage 1 Empirical Baseline Findings
- Feature extraction with Z-score standardization (`StandardScaler`) resolved gradient descent oscillations and avoided convergence warnings.
- The linear Softmax baseline achieved:
  - **Dev Accuracy:** $\approx 33.2\% - 37.5\%$
  - **Macro F1-Score:** $0.2989 - 0.3548$
  - **Macro-average ROC AUC:** $0.689 - 0.780$
  - **Inference Latency:** $\approx 25 - 35\text{ ms}$ (Satisfies $\le 50\text{ ms}$ limit)
- **Clinical Diagnosis:** High Structural Bias (*Underfitting*). While it significantly outperforms random guessing ($10\%$), the linear model severely confuses *Pasture* with *Forest* ($36\%$) and *SeaLake* ($36\%$), confirming the necessity of convolutional layers to capture spatial context.

---

### 5. Local Setup & Quick Start

#### Option A: Docker DevContainer (Recommended)
1. Install [Docker Desktop](https://www.docker.com/) and [VS Code](https://code.visualstudio.com/) with the **Dev Containers** extension.
2. Clone the repository and open it in VS Code:
   ```bash
   git clone https://github.com/dgimenezdeveloper/eurosat-ai-lab.git
   cd eurosat-ai-lab
   code .
   ```
3. Press `F1` and select **"Dev Containers: Reopen in Container"**. The container will build PyTorch, Node.js, and dependencies automatically.

#### Option B: Manual Local Setup
```bash
# 1. Clone repository & create virtual environment
git clone https://github.com/dgimenezdeveloper/eurosat-ai-lab.git
cd eurosat-ai-lab
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate

# 2. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Download dataset & generate stratified splits (80/10/10)
python scripts/setup_project.py

# 4. Launch all platform services simultaneously
python scripts/start_all.py
```

Once running, access the services:
- **React Frontend Dashboard:** [http://localhost:5173](http://localhost:5173)
- **FastAPI Documentation (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Streamlit Science Console:** [http://localhost:8501](http://localhost:8501)
- **JupyterLab Server:** [http://localhost:8888](http://localhost:8888)

---

### 6. Project Team & Academic Credentials
Developed within the **Artificial Intelligence (2026)** curriculum — *University Degree in Programming / Software Development* (Universidad Nacional Guillermo Brown - UNaB):

- **Professor:** Lic. Pablo Moreira
- **Engineering Team:**
  - **Darío Giménez** — [GitHub](https://github.com/dgimenezdeveloper) • [LinkedIn](https://www.linkedin.com/in/daseg/)
  - **Mauricio Barreras** — [GitHub](https://github.com/Mau-bar-iva) • [LinkedIn](https://www.linkedin.com/in/mauricio-barreras-235b8128a/)
  - **Federico Paál** — [GitHub](https://github.com/FedericoPaal) • [LinkedIn](https://www.linkedin.com/in/federico-paal/)
  - **Sasha Porchia** — UNaB AI Researcher

---

## 🇪🇸 Documentación en Español

### 1. Resumen Ejecutivo y Planteo del Problema
El monitoreo periódico de la cobertura del suelo y la vegetación es esencial para el pronóstico agropecuario, la planificación urbana y la gestión de recursos naturales.

**EuroSAT AI Lab** aborda este problema de visión artificial supervisada multiclase ($X \to Y$) utilizando imágenes ópticas satelitales del programa Sentinel-2 de la Agencia Espacial Europea (ESA):
- **Espacio de Entrada ($X$):** Parches satelitales de $64 \times 64$ píxeles y 3 bandas de color visible (RGB), aplanados en vectores de $12.288$ características o tratados como tensores de $64 \times 64 \times 3$.
- **Espacio de Salida ($Y$):** 10 categorías de cobertura y uso del suelo:
  1. Cultivo Anual (*AnnualCrop*)
  2. Bosque (*Forest*)
  3. Vegetación Herbácea (*HerbaceousVegetation*)
  4. Autopista / Ruta (*Highway*)
  5. Zona Industrial (*Industrial*)
  6. Pastizal / Pastura (*Pasture*)
  7. Cultivo Permanente (*PermanentCrop*)
  8. Zona Residencial (*Residential*)
  9. Río (*River*)
  10. Mar o Lago (*SeaLake*)

#### Desafíos en el Dominio Satelital
- **Ambigüedad Espectral:** Clases vegetales distintas (*Bosque*, *Pastizal*, *Vegetación Herbácea*, *Cultivo Permanente*) comparten firmas espectrales y tonalidades verdes similares en el rango visible.
- **Confusión Topológica:** La infraestructura vial (*Autopista*) y los cauces hídricos (*Río*) comparten trazos lineales continuos que requieren análisis de textura para diferenciarse.
- **Sesgo en Modelos Lineales:** El aplanado de imágenes a vectores independientes destruye las relaciones de vecindad espacial 2D, limitando la capacidad de separar clases complejas.

---

### 2. Sistema de Evaluación y Gobernanza MLOps

#### A. Partición Estratificada Fija (80 / 10 / 10)
Para evitar fuga de información y asegurar representatividad estadística, el dataset de 27.000 imágenes se divide con una semilla determinística (`seed = 42`):
- **Entrenamiento (Train - 80%, 21.600 muestras):** Destinado exclusivamente al ajuste de parámetros ($W, b$).
- **Validación (Dev - 10%, 2.700 muestras):** Utilizado para selección de arquitecturas, calibración de hiperparámetros y diagnóstico de sesgo vs. varianza.
- **Evaluación Final (Test - 10%, 2.700 muestras):** Aislado y bloqueado hasta la etapa de cierre (Etapa 5) para medir el error de generalización sin sesgo.

#### B. Métricas de Optimización y Satisfacción
- **Métrica Principal de Optimización:** **Macro F1-Score** sobre el conjunto Dev, garantizando igual ponderación para las 10 categorías sin importar variaciones de cantidad.
- **Restricciones de Satisfacción Operativa:**
  - **Latencia de Inferencia:** $\le 50\text{ ms}$ por muestra procesada en CPU estándar.
  - **Tamaño del Modelo:** $\le 100\text{ MB}$ por archivo serializado en disco.
- **Métricas Diagnósticas:** Matrices de confusión normalizadas por fila (Recall) y curvas ROC multiclase One-vs-Rest (OvR) con Macro AUC.

#### C. Bitácora de Experimentos (MLOps Ledger)
Cada corrida realizada en los Notebooks o desde el frontend web queda registrada en `artifacts/metrics/experiments_ledger.json`, documentando:
- Identificador de corrida (`run_id`, ej. `EXP-001`), fecha, hora y arquitectura.
- Hiperparámetros (Parámetro $C$, regularización $L_1/L_2$, algoritmo solver, escalado).
- Exactitud en Train/Dev, brecha de varianza (*Gap*), Macro F1 y latencia.
- Diagnóstico clínico automatizado (*Subajuste Estructural* vs. *Sobreajuste*).

---

### 3. Arquitectura y Stack Tecnológico

- **Modelado y Datos:** Python 3.10+, PyTorch 2.2, torchvision, Scikit-Learn 1.3, NumPy, Pandas, Pillow, Joblib.
- **API y Servicios:** FastAPI 0.110, Uvicorn, Pydantic v2.
- **Interfaces de Usuario:**
  - Frontend interactivo: React 19, Vite 8, Tailwind CSS v4, Base UI, Lucide Icons (Puerto 5173).
  - Consola científica: Streamlit 1.32 (Puerto 8501).
- **Entorno de Investigación:** Cuadernos JupyterLab y compatibilidad con Google Colab (Puerto 8888).
- **Contenedores:** Docker DevContainers con aceleración CUDA/CPU.

---

### 4. Hoja de Ruta en 5 Etapas

1. **Etapa 1 (Baseline Lineal - Estado Actual):** Regresión Logística multiclase (Softmax sobre 12.288 píxeles escalados con `StandardScaler`). Establece el piso de referencia superando ampliamente al azar puro ($33.2\% - 37.5\%$ de exactitud en Dev y Macro AUC de $0.689 - 0.780$), con diagnóstico de subajuste estructural.
2. **Etapa 2 (Red Densa Profunda - Siguiente):** Perceptrón Multicapa (MLP) de 3 capas ocultas con funciones de activación no lineales (GELU/ReLU), normalización por lotes (*Batch Normalization*) y regularización estocástica (*Dropout*).
3. **Etapa 3 (Red Convolucional):** CNN 2D con bloques convolucionales jerárquicos y aumento de datos (*Data Augmentation*) para modelar texturas y reducir la varianza.
4. **Etapa 4 (Modelo Generativo VAE):** Autoencoder Variacional para explorar la continuidad del espacio latente satelital.
5. **Etapa 5 (Síntesis y Evaluación Final):** Evaluación de una única pasada sobre el conjunto Test y análisis global de trade-offs.

---

### 5. Guía de Inicio Rápido

```bash
# 1. Clonar el repositorio
git clone https://github.com/dgimenezdeveloper/eurosat-ai-lab.git
cd eurosat-ai-lab

# 2. Instalar dependencias exactas
pip install -r requirements.txt

# 3. Lanzar la aplicación web de exploración e inferencia
streamlit run streamlit_app/app.py
```

---

### 6. Equipo de Desarrollo y Credenciales Académicas
Proyecto realizado en la cátedra de **Inteligencia Artificial (2026)** — *Tecnicatura Universitaria en Programación / Desarrollo de Software* (Universidad Nacional Guillermo Brown - UNaB):

- **Docente:** Lic. Pablo Moreira
- **Integrantes:**
  - **Darío Giménez** — [GitHub](https://github.com/dgimenezdeveloper) • [LinkedIn](https://www.linkedin.com/in/daseg/)
  - **Mauricio Barreras** — [GitHub](https://github.com/Mau-bar-iva) • [LinkedIn](https://www.linkedin.com/in/mauricio-barreras-235b8128a/)
  - **Federico Paál** — [GitHub](https://github.com/FedericoPaal) • [LinkedIn](https://www.linkedin.com/in/federico-paal/)
  - **Sasha Porchia** — [GitHub](https://github.com/SashaPorchia) • [LinkedIn](https://www.linkedin.com/in/sasha-porchia//)

---

## 📄 License
This project is licensed under the **MIT License**. See the repository for details.
