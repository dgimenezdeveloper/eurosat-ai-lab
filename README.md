
# EuroSAT AI Lab: Clasificación de Cobertura Terrestre sobre Imágenes Satelitales Sentinel-2

## Información Institucional y Académica
* **Institución:** Universidad Nacional Guillermo Brown (UNaB)
* **Carrera:** Tecnicatura Universitaria en Programación / Inteligencia Artificial
* **Asignatura:** Inteligencia Artificial (Ciclo Lectivo 2026)
* **Docente Titular:** Lic. Pablo Moreira
* **Integrantes del Equipo:**
  * Mauricio Barreras
  * Sasha Porchia
  * Federico Paaál
  * Darío Giménez

---
## Abrir el Notebook en Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgimenezdeveloper/eurosat-ai-lab/blob/main/notebooks/01_etapa1_eda_baseline.ipynb)

---

## 1. Definición del Problema y Justificación del Dominio

El presente proyecto aborda un problema de **aprendizaje supervisado de visión artificial multiclase** ($x \to y$), enfocado en la clasificación de patrones de cobertura del suelo (Land Use and Land Cover - LULC) a partir de imágenes satelitales del programa Sentinel-2 de la Agencia Espacial Europea (ESA).

### 1.1. Especificación del Espacio de Entrada y Salida
* **Espacio de Entrada ($X$):** Imágenes ópticas en el espectro visible (RGB) con dimensiones de $64 \times 64$ píxeles y 3 canales de color, representables como tensores $x \in \mathbb{R}^{64 \times 64 \times 3}$ o vectores planos de $12.288$ características numéricas.
* **Espacio de Salida ($Y$):** Etiqueta categórica discreta $y \in \{0, 1, \dots, 9\}$ correspondiente a 10 clases de uso de suelo:
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

### 1.2. Desafíos de la Tarea en Visión Artificial
* **Ambigüedad Espectral:** Múltiples clases de vegetación (*Forest*, *Pasture*, *HerbaceousVegetation*, *PermanentCrop*) comparten firmas espectrales y tonalidades verdes similares en el rango visible.
* **Ambigüedad Topológica:** Estructuras lineales artificiales (*Highway*) y naturales (*River*) poseen geometrías continuas comparables que requieren extracción de contexto textural para su discriminación.
* **Invarianza Espacial y Complejidad:** La resolución de $64 \times 64$ píxeles obliga a modelar dependencias jerárquicas locales (bordes, esquinas y texturas compuestas) que un modelo lineal no puede capturar.

---

## 2. Estrategia Pragmática de Partición y Evaluación

Siguiendo las directrices metodológicas de la asignatura, el diseño experimental se estructura para garantizar rigor estadístico y prevenir la fuga de información (*data leakage*).

### 2.1. Partición de Datos Estratificada (80 / 10 / 10)
El conjunto total de 27.000 imágenes se divide de forma pseudoaleatoria y estratificada fijando una semilla determinística (`seed = 42`):

* **Conjunto de Entrenamiento (Train - 80%, 21.600 muestras):** Destinado exclusivamente a la optimización de parámetros internos (pesos $W$ y sesgos $b$) mediante algoritmos basados en gradiente.
* **Conjunto de Desarrollo / Validación (Dev - 10%, 2.700 muestras):** Utilizado como banco de iteración para la selección de arquitecturas, calibración de hiperparámetros (tasas de aprendizaje, regularización, tamaño de lote) y diagnóstico clínico de errores (sesgo vs. varianza).
* **Conjunto de Prueba (Test - 10%, 2.700 muestras):** Mantenido en aislamiento durante todo el ciclo de investigación. Se evaluará **una única vez en la Etapa 5** para estimar el error de generalización no sesgado.

```
Total: 27.000 Imágenes Satelitales
├── Train Set (80%): 21.600 muestras [Ajuste de Parámetros]
├── Dev Set   (10%):  2.700 muestras [Ajuste de Hiperparámetros y Diagnóstico]
└── Test Set  (10%):  2.700 muestras [Evaluación Final Única - Etapa 5]
```

---

## 3. Sistema de Métricas de Evaluación

Para evitar optimizaciones conflictivas, se establece una clara distinción entre la métrica única de optimización y las restricciones de satisfacción operativas.

### 3.1. Métrica de Optimización (Single-Number Metric): Macro F1-Score
Dado que la exactitud global (*Accuracy*) puede ocultar debilidades en categorías críticas o con ligeras variaciones de balance, se adopta el **Macro F1-Score** sobre el conjunto Dev como criterio principal de selección de modelos:

$$\text{Macro F1} = \frac{1}{K}\sum_{k=1}^{K} F1_k, \quad \text{donde } F1_k = 2 \cdot \frac{\text{Precisión}_k \cdot \text{Recall}_k}{\text{Precisión}_k + \text{Recall}_k}$$

### 3.2. Métricas de Satisfacción (Satisficing Metrics)
Restricciones de viabilidad de ingeniería requeridas para el despliegue del sistema:
* **Latencia de Inferencia:** $\le 50\text{ ms}$ por imagen procesada en arquitectura CPU estándar.
* **Tamaño del Modelo en Disco:** $\le 100\text{ MB}$ por artefacto serializado.

### 3.3. Métricas Diagnósticas Secundarias
* **Matriz de Confusión Normalizada por Filas:** Permite identificar la tasa de acierto directo (*Recall*) por clase en la diagonal principal y cuantificar los patrones sistemáticos de confusión cruzada.
* **Curvas ROC Multiclase One-vs-Rest (OvR) y Macro AUC:** Evalúan la capacidad discriminativa del modelo a través de todos los umbrales de decisión $\theta \in [0, 1]$.

---

## 4. Marco Teórico y Protocolo de Diagnóstico de Errores

El proyecto utiliza la descomposición formal del error esperado de generalización para guiar las modificaciones arquitectónicas:

$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Sesgo}^2 + \text{Varianza} + \sigma_\varepsilon^2$$

Donde:
* $\text{Sesgo}^2 = (\mathbb{E}[\hat{f}(x)] - f(x))^2$: Error por incapacidad estructural de la hipótesis para capturar la función subyacente (*Underfitting*).
* $\text{Varianza} = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]$: Sensibilidad del modelo al ruido específico de la muestra de entrenamiento (*Overfitting*).
* $\sigma_\varepsilon^2$: Ruido irreducible inherente a los datos de teledetección.

### 4.1. Protocolo Diagnóstico Integrado
Se monitorean los errores empíricos y la brecha de varianza:

$$\text{Error}_{\text{Train}} = 1 - \text{Accuracy}_{\text{Train}}$$
$$\text{Error}_{\text{Dev}} = 1 - \text{Accuracy}_{\text{Dev}}$$
$$\text{Gap} = \text{Error}_{\text{Dev}} - \text{Error}_{\text{Train}}$$

* **Diagnóstico de Sesgo Alto ($\text{Error}_{\text{Train}} \gg 0$):**  
  * *Acción:* Aumentar capacidad de la red (añadir capas o neuronas), sustituir transformaciones lineales por funciones de activación no lineales (GELU, ReLU), reducir regularización excesiva.
* **Diagnóstico de Varianza Alta ($\text{Error}_{\text{Dev}} \gg \text{Error}_{\text{Train}}$ / $\text{Gap} > 10\%$):**  
  * *Acción:* Incorporar regularización $L_2$ (*Weight Decay*), regularización estocástica (*Dropout*), normalización interna (*Batch Normalization*), aumento de datos (*Data Augmentation*) y detención temprana (*Early Stopping*).
* **Diagnóstico de Discrepancia de Distribución (*Data Mismatch*):**  
  * Se evalúa mediante la relación $\text{Error}_{\text{Dev-Train}} > \text{Error}_{\text{Train}}$. En caso de presentarse, la solución consiste en realinear los datos representativos de producción.

---

## 5. Evolución Progresiva del Trabajo Práctico por Etapas

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                        MAPA DE EVOLUCIÓN EXPERIMENTAL                          │
├─────────────────┬──────────────────────────────────┬───────────────────────────┤
│ Etapa           │ Modelo / Metodología             │ Objetivo Teórico          │
├─────────────────┼──────────────────────────────────┼───────────────────────────┤
│ 1. Baseline     │ Regresión Logística Multiclase   │ Establecer piso lineal y  │
│    (Completada) │ (Softmax sobre 12.288 píxeles)   │ verificar pipeline.       │
├─────────────────┼──────────────────────────────────┼───────────────────────────┤
│ 2. MLP Profundo │ Red Densa (3+ capas ocultas,     │ Quebrar el sesgo lineal   │
│    (Siguiente)  │ GELU, Dropout, BatchNorm)        │ con no linealidad.        │
├─────────────────┼──────────────────────────────────┼───────────────────────────┤
│ 3. Convolucional│ CNN 2D (Bloques Conv + MaxPool)  │ Extraer jerarquías        │
│                 │ + Data Augmentation              │ espaciales y texturas.    │
├─────────────────┼──────────────────────────────────┼───────────────────────────┤
│ 4. VAE          │ Variational Autoencoder          │ Modelar espacio latente e │
│    (Opcional)   │ (Encoder-Decoder probabilístico) │ interpolación generativa. │
├─────────────────┼──────────────────────────────────┼───────────────────────────┤
│ 5. Síntesis     │ Evaluación final en Test Set     │ Comparación global, trade-│
│    (Cierre)     │ (Evaluación única)               │ off y análisis ético.     │
└─────────────────┴──────────────────────────────────┴───────────────────────────┘
```

### 5.1. Etapa 1: Análisis Exploratorio y Modelo Baseline (Estado Actual)
* **Preprocesamiento y Normalización:**  
  El análisis exploratorio de los canales RGB evidenció medias desplazadas ($\mu \approx 80-100$) y dispersión amplia ($\sigma \approx 35-50$). Se aplicó `StandardScaler` sobre el vector aplanado de $12.288$ características para acondicionar la superficie de error y evitar oscilaciones patológicas en el optimizador L-BFGS.
* **Resultados Empíricos del Baseline:**
  * Exactitud en Train: $\approx 85.2\%$ (sobre subconjunto muestral)
  * Exactitud en Dev: $35.2\% - 37.5\%$
  * Macro F1-Score en Dev: $0.3306 - 0.3548$
  * Latencia de Inferencia: $\approx 30.37\text{ ms}$
* **Diagnóstico de Etapa 1:**  
  El modelo lineal supera al azar puro ($10\%$), validando la integridad de los datos. Sin embargo, sufre de un **Sesgo Alto estructural**: una combinación lineal $\sum w_i x_i + b$ colapsa matemáticamente y carece de noción de vecindad espacial, confundiendo severamente cubiertas vegetales homogéneas (*Pasture* vs. *Forest*) y patrones lineales (*Highway* vs. *River*).

### 5.2. Etapa 2: Análisis Cualitativo de Errores y Perceptrón Multicapa (MLP)
* **Auditoría de 50 Errores:** Clasificación taxonómica de las instancias peor clasificadas por el Baseline para identificar ambigüedades espectrales y de textura.
* **Arquitectura de Red Densa:** Implementación de un modelo de 3 capas ocultas ($512 \to 256 \to 128$ neuronas).
* **Funciones de Activación No Lineales:** Utilización de **GELU** (*Gaussian Error Linear Unit*) y **ReLU** para evitar el colapso lineal y mitigar el desvanecimiento de gradientes característico de funciones sigmoideas y tanh en capas profundas.
* **Control de Varianza:** Incorporación de **Batch Normalization** entre capas densas y **Dropout** estocástico ($p \in [0.2, 0.4]$).

### 5.3. Etapa 3: Redes Neuronales Convolucionales (CNN)
* **Arquitectura Convolucional:** Diseño de una red con 3 bloques jerárquicos (`Conv2D` con kernels $3 \times 3$ + `BatchNorm2d` + `ReLU` + `MaxPool2d`).
* **Invarianza Espacial y Reducción de Parámetros:** Explotación de la correlación local de píxeles mediante campos receptivos, reduciendo drásticamente la cantidad de parámetros frente a capas densas equivalentes.
* **Data Augmentation:** Aplicación de transformaciones afines aleatorias (rotaciones ortogonales, reflejos horizontales/verticales) para enriquecer la distribución de entrenamiento y suprimir la varianza.

### 5.4. Etapa 4: Modelado Generativo Latente con VAE (Opcional)
* **Arquitectura Variacional:** Implementación de un codificador que mapea imágenes a los parámetros de una distribución normal multivariada ($\mu_z, \log \sigma_z^2$) y un decodificador reconstructivo.
* **Función de Pérdida VAE:** Combinación de error de reconstrucción (MSE / BCE) y divergencia de Kullback-Leibler ($\mathcal{D}_{\text{KL}}$) para regularizar la continuidad del espacio latente.
* **Análisis de Espacio Latente:** Visualización de clústeres mediante reducción dimensional (t-SNE/PCA) e interpolación lineal entre clases.

### 5.5. Etapa 5: Evaluación Final y Síntesis Comparativa
* **Desbloqueo de Test Set:** Ejecución de una única pasada sobre las 2.700 imágenes finales.
* **Tabla Maestra Comparativa:** Análisis de trade-off entre número de parámetros entrenables, latencia de inferencia, tamaño en disco y Macro F1 en Train, Dev y Test.
* **Análisis Ético y Limitaciones:** Evaluación de sesgos geográficos del satélite Sentinel-2, variabilidad estacional y límites de resolución espacial para la toma de decisiones agrícolas o urbanas.

---

## 6. Arquitectura de Repositorio y Trazabilidad MLOps

El repositorio implementa una separación estricta entre el entorno de experimentación científica y la capa de servicio productiva:

```text
euro-sat-ai-lab/
├── .devcontainer/              # Entorno reproducible Docker (PyTorch + CUDA + Node.js)
├── data/
│   ├── raw/                    # 27.000 imágenes Sentinel-2 organizadas por clase
│   └── processed/              # splits.json (partición fija Train/Dev/Test)
├── notebooks/                  # Cuadernos académicos explicativos (Fuente de Verdad)
│   ├── 01_etapa1_eda_baseline.ipynb
│   ├── 02_etapa2_mlp_regularizacion.ipynb
│   ├── 03_etapa3_cnn_augmentation.ipynb
│   ├── 04_etapa4_vae_latente.ipynb
│   └── 05_etapa5_sintesis_comparativa.ipynb
├── src/                        # Módulos Python empaquetados y reutilizables
│   ├── config.py               # Constantes, semillas y mapeo de clases (EN/ES)
│   ├── data/                   # Carga de imágenes y pipelines de transformación
│   ├── models/                 # Arquitecturas: Baseline, MLP, CNN, VAE
│   ├── training/               # Bucles de optimización, early stopping y tracker
│   ├── evaluation/             # Matrices de confusión, curvas ROC y diagnóstico
│   └── api/                    # Servicio de inferencia y telemetría (FastAPI)
├── artifacts/                  # Almacenamiento inmutable de resultados empíricos
│   ├── models/                 # Pesos y modelos serializados (.joblib / .pt)
│   └── metrics/                # Bitácora MLOps (experiments_ledger.json, summary)
├── streamlit_app/              # Interfaz de exploración científica interactiva
└── frontend/                   # Dashboard de presentación y monitoreo (React + Vite)
```

### 6.1. Bitácora Inmutable de Experimentos (MLOps Ledger)
Cada ejecución de entrenamiento genera un registro estructurado en `artifacts/metrics/experiments_ledger.json` que documenta:
* Identificador único de corrida (`run_id`, ej. `EXP-001`).
* Vector de hiperparámetros ($C$, tipo de regularización $L_1/L_2$, algoritmo solver, método de escalado, número de iteraciones).
* Métricas obtenidas ($\text{Accuracy}_{\text{Train}}$, $\text{Accuracy}_{\text{Dev}}$, $\text{Gap}$, $\text{Macro F1}$, latencia).
* Diagnóstico clínico automatizado según las reglas formales de la Clase 3.
* Hipótesis o justificación cualitativa registrada por el experimentador.

---

## 7. Protocolo de Reproducibilidad y Ejecución

1. **Entorno Docker Aislado:**  
   El proyecto utiliza un contenedor con Python 3.10, PyTorch 2.2.1 con aceleración CUDA/CPU y Node.js 20, asegurando la consistencia de versiones en cualquier sistema operativo.
2. **Determinismo:**  
   Todas las operaciones estocásticas (particionado, inicialización de tensores, generadores de números aleatorios) utilizan una semilla fija (`seed = 42`).
3. **Persistencia:**  
   Los datos crudos de EuroSAT se descargan una única vez y se conservan desacoplados del control de versiones mediante `.gitignore` para optimizar el almacenamiento del repositorio.
