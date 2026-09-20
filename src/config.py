# src/config.py
import os
import sys
import torch

IS_COLAB = "google.colab" in sys.modules
BASE_DIR = "/content/eurosat-ai-lab" if IS_COLAB else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SEED = 42
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "eurosat", "2750")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
SPLITS_PATH = os.path.join(DATA_PROCESSED_DIR, "splits.json")

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")