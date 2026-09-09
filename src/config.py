import os
import torch

SEED = 42
DATA_RAW_DIR = "./data/raw/eurosat/2750"
DATA_PROCESSED_DIR = "./data/processed"
ARTIFACTS_DIR = "./artifacts"

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
