import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
import joblib

# Asegurar que reconozca 'src' sin importar desde dónde se ejecute
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torchvision.datasets import ImageFolder
from src.config import DATA_RAW_DIR, SPLITS_PATH, ARTIFACTS_DIR, CLASS_NAMES, CLASS_NAMES_ES

def main():
    print("=== Extrayendo 50 errores cualitativos del Baseline (Dev Set) ===")
    
    baseline_path = os.path.join(ARTIFACTS_DIR, "models", "baseline_logreg.joblib")
    if not os.path.exists(baseline_path):
        print("[AVISO] No se encontró baseline_logreg.joblib. Entrenando baseline rápido...")
        os.system("python scripts/train_baseline.py")

    artifact = joblib.load(baseline_path)
    model = artifact["model"]
    scaler = artifact["scaler"]

    with open(SPLITS_PATH, "r") as f:
        splits = json.load(f)
    dev_indices = splits["dev_indices"]

    dataset = ImageFolder(root=DATA_RAW_DIR)

    # Extraer características del dev set
    print("Extrayendo características de validación (Dev Set)...")
    X_dev, y_dev = [], []
    for idx in dev_indices:
        img, target = dataset[idx]
        X_dev.append(np.array(img, dtype=np.float32).flatten())
        y_dev.append(target)

    X_dev = np.array(X_dev)
    y_dev = np.array(y_dev)

    X_dev_scaled = scaler.transform(X_dev)
    probs = model.predict_proba(X_dev_scaled)
    preds = np.argmax(probs, axis=1)

    # Identificar índices mal clasificados
    error_mask = preds != y_dev
    error_dev_indices = [dev_indices[i] for i in range(len(dev_indices)) if error_mask[i]]
    error_real = y_dev[error_mask]
    error_preds = preds[error_mask]
    error_probs = probs[error_mask]

    print(f"Total en Dev: {len(dev_indices)} | Errores encontrados: {len(error_dev_indices)} ({len(error_dev_indices)/len(dev_indices)*100:.2f}%)")

    export_dir = os.path.join(ARTIFACTS_DIR, "error_samples")
    os.makedirs(export_dir, exist_ok=True)

    errors_records = []
    count = 0

    for i, global_idx in enumerate(error_dev_indices):
        if count >= 50:
            break
        
        real_idx = error_real[i]
        pred_idx = error_preds[i]
        real_name = CLASS_NAMES[real_idx]
        pred_name = CLASS_NAMES[pred_idx]
        conf = float(error_probs[i][pred_idx])
        real_conf = float(error_probs[i][real_idx])

        # Diagnóstico clínico para la cátedra
        if pred_name in ["River", "Highway"] and real_name in ["River", "Highway"]:
            category = "Ambigüedad Topológica (Estructuras lineales)"
            justification = "El modelo lineal carece de contexto espectral para diferenciar agua de asfalto en geometrías lineales continuas."
        elif pred_name in ["Forest", "Pasture", "HerbaceousVegetation"] and real_name in ["Pasture", "Forest", "HerbaceousVegetation"]:
            category = "Sesgo Alto (Firma Espectral Clorofílica)"
            justification = "Al no poseer filtros convolucionales de textura, promedia tonos verdes sin distinguir densidad foliar."
        elif pred_name in ["AnnualCrop", "PermanentCrop"] and real_name in ["PermanentCrop", "AnnualCrop"]:
            category = "Sesgo Alto (Falta de Consciencia de Patrón Espacial)"
            justification = "Incapaz de detectar los surcos geométricos de plantaciones arbóreas vs. parcelas de arado temporal."
        elif conf > 0.60:
            category = "Varianza / Confianza Engañosa del Hiperplano"
            justification = "El hiperplano asigna alta probabilidad en una región del espacio donde los píxeles aplanados colapsan lejos del margen."
        else:
            category = "Sesgo Alto (Incapacidad Representacional)"
            justification = "Incapacidad intrínseca de una combinación lineal simple W*x + b para modelar texturas multiespectrales."

        img, _ = dataset[global_idx]
        img_filename = f"error_{count+1:02d}_{real_name}_pred_{pred_name}.png"
        img.save(os.path.join(export_dir, img_filename))

        errors_records.append({
            "error_id": count + 1,
            "dataset_index": int(global_idx),
            "image_file": img_filename,
            "real_class": real_name,
            "real_class_es": CLASS_NAMES_ES[real_name],
            "predicted_class": pred_name,
            "predicted_class_es": CLASS_NAMES_ES[pred_name],
            "confidence_predicted": round(conf * 100, 2),
            "confidence_real": round(real_conf * 100, 2),
            "error_category": category,
            "clinical_justification": justification
        })
        count += 1

    metadata_path = os.path.join(export_dir, "errors_50_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(errors_records, f, indent=2, ensure_ascii=False)

    print(f"✓ Metadatos y 50 imágenes guardados con éxito en {export_dir}")

if __name__ == "__main__":
    main()
