"""Copia una muestra fija de imágenes de Dev y Test (20 por clase) a streamlit_app/samples/.

El dataset crudo (2 GB) no se sube a GitHub, así que en Streamlit Community Cloud la
inferencia en vivo usa estas imágenes. Se copian los .jpg originales sin modificar.

Uso: python scripts/export_demo_samples.py
"""
import json
import os
import random
import shutil
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
from src.config import DATA_RAW_DIR, SPLITS_PATH  # noqa: E402

PER_CLASS = 20
OUT_DIR = os.path.join(BASE_DIR, "streamlit_app", "samples")


def main():
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(root=DATA_RAW_DIR)
    with open(SPLITS_PATH, encoding="utf-8") as f:
        splits = json.load(f)
    rng = random.Random(42)
    for split in ("dev", "test"):
        for target, cls in enumerate(dataset.classes):
            pool = [i for i in splits[f"{split}_indices"] if dataset.targets[i] == target]
            dest = os.path.join(OUT_DIR, split, cls)
            os.makedirs(dest, exist_ok=True)
            for i in rng.sample(pool, PER_CLASS):
                src = dataset.samples[i][0]
                shutil.copy(src, os.path.join(dest, os.path.basename(src)))
    print(f"[OK] {PER_CLASS} imágenes por clase de Dev y Test copiadas en {OUT_DIR}")


if __name__ == "__main__":
    main()
