import os
import json
import time
import numpy as np
import joblib
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torchvision.datasets import ImageFolder

DATA_DIR = "./data/raw/eurosat/2750"
SPLITS_PATH = "./data/processed/splits.json"
ARTIFACTS_DIR = "./artifacts"

print("=== [1/4] Cargando imágenes y particiones ===")
with open(SPLITS_PATH, "r") as f:
    splits = json.load(f)

classes = splits["classes"]
train_idx = splits["train_indices"]
dev_idx = splits["dev_indices"]

dataset = ImageFolder(root=DATA_DIR)

def extract_features(indices, max_samples=4000):
    X, y = [], []
    sub = indices[:max_samples]
    for idx in sub:
        img, target = dataset[idx]
        X.append(np.array(img, dtype=np.float32).flatten())
        y.append(target)
    return np.array(X), np.array(y)

print("=== [2/4] Aplanando píxeles (64x64x3 = 12.288 features) y Escalando ===")
X_train, y_train = extract_features(train_idx, max_samples=4000)
X_dev, y_dev = extract_features(dev_idx, max_samples=1000)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_dev_scaled = scaler.transform(X_dev)

print("=== [3/4] Entrenando Regresión Logística (Baseline)... ===")
start = time.time()
clf = LogisticRegression(max_iter=150, C=1.0, solver='lbfgs')
clf.fit(X_train_scaled, y_train)
train_time = time.time() - start

print(f"✓ Entrenamiento completado en {train_time:.2f} segundos.")

print("=== [4/4] Evaluando en conjunto Dev (Validación) y exportando ===")
y_pred = clf.predict(X_dev_scaled)
acc = accuracy_score(y_dev, y_pred)
macro_f1 = f1_score(y_dev, y_pred, average='macro')
latency_ms = round((train_time / len(X_dev)) * 1000, 2)

print(f" -> Exactitud (Accuracy): {acc * 100:.2f}%")
print(f" -> Macro F1-Score:      {macro_f1:.4f}")

# Guardar modelo y escalador
joblib.dump({"model": clf, "scaler": scaler, "classes": classes}, f"{ARTIFACTS_DIR}/models/baseline_logreg.joblib")

# Actualizar models_summary.json
summary = [
    {
        "name": "Baseline (LogReg)",
        "version": "v1.0.0",
        "accuracy": round(acc * 100, 2),
        "f1": round(macro_f1, 4),
        "latency": latency_ms,
        "status": "Production"
    }
]
with open(f"{ARTIFACTS_DIR}/metrics/models_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Guardar matriz de confusión real
cm = confusion_matrix(y_dev, y_pred).tolist()
with open(f"{ARTIFACTS_DIR}/metrics/confusion_matrix_cnn.json", "w") as f:
    json.dump(cm, f)

print("\n🚀 ¡Métricas reales exportadas! Recarga el navegador en localhost:5173 para ver los cambios.")
