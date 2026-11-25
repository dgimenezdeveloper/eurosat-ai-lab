"""Reconstruye el StandardScaler con el que el notebook entregado entrenó el MLP (Etapa 2).

El notebook (celda "3. Modelo Baseline") ajusta un StandardScaler sobre las primeras
4.000 imágenes de Train aplanadas, pero no lo guarda. Este script repite ese mismo
cálculo (determinístico) para que la app de Streamlit pueda usar mlp_regularized.pt
de la corrida del 21/11 en inferencia en vivo.

Uso: python scripts/build_mlp_scaler.py
"""
import json
import os
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler
from torchvision.datasets import ImageFolder

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
from src.config import DATA_RAW_DIR, SPLITS_PATH  # noqa: E402

MAX_TRAIN_SAMPLES = 4000  # mismo valor que el notebook
OUT_PATH = os.path.join(BASE_DIR, "artifacts", "models", "mlp_standard_scaler.npz")


def main():
    dataset = ImageFolder(root=DATA_RAW_DIR)
    with open(SPLITS_PATH, encoding="utf-8") as f:
        train_idx = json.load(f)["train_indices"][:MAX_TRAIN_SAMPLES]
    X = np.array([np.array(dataset[i][0], dtype=np.float32).flatten() for i in train_idx])
    scaler = StandardScaler().fit(X)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(OUT_PATH, mean=scaler.mean_, scale=scaler.scale_)
    print(f"[OK] StandardScaler ({X.shape[0]} imágenes de Train) guardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
