# TP Final IA 2026 — Clasificación de Cobertura Terrestre con EuroSAT RGB

**Universidad Nacional Guillermo Brown — Inteligencia Artificial, 2° cuatrimestre 2026**
Docente: Pablo Moreira · Grupo 5: Barreras, Porchia, Paal, Gimenez

Desarrollo progresivo de un sistema de clasificación de imágenes satelitales Sentinel-2 (10 clases, 27.000 parches RGB de 64×64). El sistema pasa por un baseline lineal, un MLP regularizado, una CNN y un VAE generativo, y cierra con la evaluación final en Test y un análisis ético.

## Contenido de la entrega

| Archivo / carpeta | Contenido |
| :--- | :--- |
| `TP_IA2026_BARRERAS_PORCHIA_PAAL_GIMENEZ_GRUPO5.ipynb` | Notebook principal con las 5 etapas y todas las salidas de la ejecución final |
| `requirements.txt` | Librerías y versiones exactas del entorno en que se ejecutó el notebook |
| `src/` | Módulos auxiliares que importa el notebook (`config.py`: rutas, semilla, nombres de clases; `training/tracker.py`: registro de experimentos) |
| `data/processed/splits.json` | Partición estratificada congelada 80/10/10 (`SEED = 42`) con los índices de Train, Dev y Test |
| `resultados/figuras/` | Todos los gráficos del notebook, exportados automáticamente en orden de aparición |
| `resultados/models_comparison.csv` | Tabla comparativa de todos los modelos: arquitectura, configuración, parámetros, métricas en Dev y Test, tiempo |
| `resultados/etapa5_resultados.json` | Métricas de la Etapa 5: Test por clase, matriz de confusión, latencia, intervalo bootstrap, subgrupos |

## Cómo reproducir

### Opción A — Google Colab (recomendada)

1. Abrir el notebook en Colab (`Archivo → Subir notebook`).
2. `Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4)`.
3. `Entorno de ejecución → Ejecutar todas`.

La primera celda clona el repositorio del proyecto (que incluye `src/` y la partición), descarga EuroSAT (~90 MB) y, si se autoriza Google Drive, lo guarda en caché para no volver a bajarlo en las sesiones siguientes. Con GPU T4, la ejecución completa tarda unos 30 minutos.

### Opción B — Entorno local

Requiere Python 3.10 o superior. Sin GPU funciona, pero las Etapas 3 y 4 tardan varias horas.

```bash
cd TP_IA2026_BARRERAS_PORCHIA_PAAL_GIMENEZ_EuroSAT
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyter

# Descarga de EuroSAT RGB a data/raw/eurosat/2750
python -c "from torchvision.datasets import EuroSAT; EuroSAT(root='data/raw', download=True)"

jupyter notebook TP_IA2026_BARRERAS_PORCHIA_PAAL_GIMENEZ_GRUPO5.ipynb
```

## Organización del notebook

| Etapa | Contenido | Modelos |
| :---: | :--- | :--- |
| 1 | EDA, partición estratificada, métrica de número único (Macro F1), baseline | Regresión logística (C=1.0 y C=0.01) |
| 2 | Análisis manual de 50 errores, MLP, regularización (con y sin) | MLP Vanilla y MLP con BatchNorm + Dropout + L2 + Early Stopping |
| 3 | CNN de 3 bloques, Data Augmentation, End-to-End vs. secuencial, atribución de errores | CNN (3 variantes), ResNet-18 + LogReg, backbone propio + LogReg |
| 4 | Modelo generativo, espacio latente (PCA/t-SNE, interpolación, manipulación), métricas | VAE convolucional (latente de 64 dimensiones) |
| 5 | Tabla comparativa, evaluación única en Test, desempeño por subgrupos, análisis ético, conclusiones | — |

## Decisiones de reproducibilidad

* **Semilla fija** (`SEED = 42`) en NumPy, PyTorch y scikit-learn, y **partición congelada** en `splits.json`.
* **Test se usa una sola vez**, en la Etapa 5. El modelo campeón se elige antes, sólo con el Macro F1 en Dev.
* **Las cifras de las conclusiones se generan desde el código** con las métricas de la ejecución. En GPU, algunos resultados pueden variar unas décimas entre corridas por no determinismo de cuDNN.
* Las Etapas 1 y 2 usan 4.000 imágenes de Train y 1.000 de Dev, por el costo en memoria de aplanar las imágenes. Desde la Etapa 3 se usa el dataset completo. La Etapa 3 incluye una CNN de control entrenada con las mismas 4.000 imágenes para que la comparación sea justa.
