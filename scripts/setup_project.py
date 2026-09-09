# scripts/setup_project.py
import os
import ssl
import json
import urllib.request
import torch
import torchvision
from torchvision.datasets import EuroSAT
from sklearn.model_selection import train_test_split

# --- SOLUCIÓN AL ERROR SSL DEL SERVIDOR DE EUROSAT (DFKI) ---
ssl._create_default_https_context = ssl._create_unverified_context

SEED = 42
DATA_RAW_DIR = "./data/raw"
DATA_PROCESSED_DIR = "./data/processed"
ARTIFACTS_DIR = "./artifacts"

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

def main():
    print("=== [1/3] Creando directorios del proyecto ===")
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    os.makedirs(f"{ARTIFACTS_DIR}/models", exist_ok=True)
    os.makedirs(f"{ARTIFACTS_DIR}/metrics", exist_ok=True)
    os.makedirs(f"{ARTIFACTS_DIR}/error_samples", exist_ok=True)

    print("\n=== [2/3] Descargando EuroSAT RGB (~2GB, 27.000 imágenes) ===")
    # Con el contexto SSL parchado, la descarga no fallará
    dataset = EuroSAT(root=DATA_RAW_DIR, download=True)
    print("✓ Descarga y extracción completadas.")

    print("\n=== [3/3] Generando Partición Estratificada Fija (80/10/10) ===")
    targets = dataset.targets
    indices = list(range(len(targets)))

    # 80% Train, 20% Temp
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.20, random_state=SEED, stratify=targets
    )

    # 10% Dev, 10% Test
    temp_targets = [targets[i] for i in temp_idx]
    dev_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=SEED, stratify=temp_targets
    )

    splits = {
        "classes": dataset.classes,
        "train_indices": train_idx,
        "dev_indices": dev_idx,
        "test_indices": test_idx
    }

    split_file = os.path.join(DATA_PROCESSED_DIR, "splits.json")
    with open(split_file, "w") as f:
        json.dump(splits, f, indent=2)

    # Plantilla inicial de modelos para el dashboard
    summary_file = os.path.join(ARTIFACTS_DIR, "metrics", "models_summary.json")
    if not os.path.exists(summary_file):
        initial_summary = [
            {
                "name": "Baseline (LogReg)",
                "version": "v1.0.0",
                "accuracy": 0.0,
                "f1": 0.0,
                "latency": 0,
                "status": "Pending Training"
            }
        ]
        with open(summary_file, "w") as f:
            json.dump(initial_summary, f, indent=2)

    print("\n¡Configuración inicial completada con éxito!")
    print(f" - Train: {len(train_idx)} | Dev: {len(dev_idx)} | Test: {len(test_idx)}")
    print(f" - Metadatos guardados en: {split_file}")

if __name__ == "__main__":
    main()