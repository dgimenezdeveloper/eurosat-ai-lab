import io
import json
import os
import time
import warnings
from typing import Dict, List
import numpy as np
import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from torchvision.datasets import ImageFolder
from src.config import ARTIFACTS_DIR, CLASS_NAMES, CLASS_NAMES_ES, DATA_RAW_DIR, DATA_PROCESSED_DIR
from src.training.tracker import log_experiment, LEDGER_PATH

# Silenciar advertencias de convergencia en logs de API
warnings.filterwarnings("ignore", category=ConvergenceWarning)

app = FastAPI(title="EuroSAT AI Lab API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

baseline_data = None

def get_baseline():
    global baseline_data
    if baseline_data is None:
        path = os.path.join(ARTIFACTS_DIR, "models", "baseline_logreg.joblib")
        if os.path.exists(path):
            baseline_data = joblib.load(path)
    return baseline_data

class RetrainRequest(BaseModel):
    C: float = 1.0
    max_iter: int = 250
    penalty: str = "l2"
    solver: str = "lbfgs"
    scaler_type: str = "standard"
    max_samples: int = 3000
    notes: str = ""

@app.get("/health")
def health():
    return {"status": "healthy", "device": "cpu", "model_loaded": get_baseline() is not None}

@app.get("/classes")
def classes():
    return CLASS_NAMES

@app.get("/models")
def models():
    path = os.path.join(ARTIFACTS_DIR, "metrics", "models_summary.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

@app.get("/experiments")
def get_experiments():
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, "r") as f:
            return json.load(f)
    return []

@app.get("/metrics")
def metrics():
    cm_path = os.path.join(ARTIFACTS_DIR, "metrics", "confusion_matrix_cnn.json")
    cm = []
    if os.path.exists(cm_path):
        with open(cm_path, "r") as f:
            cm = json.load(f)
    return {"confusion_matrix": cm}

@app.post("/train/baseline")
def retrain_baseline(req: RetrainRequest):
    global baseline_data
    splits_path = os.path.join(DATA_PROCESSED_DIR, "splits.json")
    if not os.path.exists(splits_path):
        raise HTTPException(status_code=400, detail="splits.json no encontrado.")

    with open(splits_path, "r") as f:
        splits = json.load(f)

    train_idx = splits["train_indices"][:req.max_samples]
    dev_idx = splits["dev_indices"][:1000]

    dataset = ImageFolder(root=DATA_RAW_DIR)

    def extract(indices):
        X, y = [], []
        for idx in indices:
            img, target = dataset[idx]
            X.append(np.array(img, dtype=np.float32).flatten())
            y.append(target)
        return np.array(X), np.array(y)

    X_train, y_train = extract(train_idx)
    X_dev, y_dev = extract(dev_idx)

    if req.scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)

    actual_solver = req.solver
    if req.penalty == "l1" and actual_solver == "lbfgs":
        actual_solver = "saga"

    start = time.time()
    clf = LogisticRegression(
        max_iter=req.max_iter,
        C=req.C,
        penalty=req.penalty,
        solver=actual_solver,
        tol=1e-3
    )
    clf.fit(X_train_scaled, y_train)
    train_time = time.time() - start

    y_train_pred = clf.predict(X_train_scaled)
    y_dev_pred = clf.predict(X_dev_scaled)

    train_acc = round(float(accuracy_score(y_train, y_train_pred)) * 100, 2)
    dev_acc = round(float(accuracy_score(y_dev, y_dev_pred)) * 100, 2)
    macro_f1 = round(float(f1_score(y_dev, y_dev_pred, average="macro")), 4)
    latency = round((train_time / len(X_dev)) * 1000, 2)

    baseline_data = {"model": clf, "scaler": scaler, "classes": CLASS_NAMES}
    joblib.dump(baseline_data, os.path.join(ARTIFACTS_DIR, "models", "baseline_logreg.joblib"))

    cm = confusion_matrix(y_dev, y_dev_pred).tolist()
    with open(os.path.join(ARTIFACTS_DIR, "metrics", "confusion_matrix_cnn.json"), "w") as f:
        json.dump(cm, f)

    log_entry = log_experiment(
        model_name=f"Baseline (Regresión Logística {req.penalty.upper()})",
        hyperparameters={
            "C": req.C,
            "penalty": req.penalty,
            "solver": actual_solver,
            "scaler": req.scaler_type,
            "max_iter": req.max_iter
        },
        metrics={
            "train_accuracy": train_acc,
            "dev_accuracy": dev_acc,
            "macro_f1": macro_f1,
            "latency_ms": latency
        },
        dataset_info={"train_samples": len(train_idx), "dev_samples": len(dev_idx)},
        notes=req.notes
    )

    return {
        "status": "success",
        "run_id": log_entry["run_id"],
        "accuracy": dev_acc,
        "f1": macro_f1,
        "diagnosis": log_entry["diagnosis"],
        "train_time_sec": round(train_time, 2)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    b = get_baseline()
    if b is None:
        raise HTTPException(status_code=503, detail="Modelo no entrenado aún.")

    start = time.perf_counter()
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB").resize((64, 64))
    
    flat = np.array(image, dtype=np.float32).flatten().reshape(1, -1)
    scaled = b["scaler"].transform(flat)
    
    probs = b["model"].predict_proba(scaled)[0]
    latency = round((time.perf_counter() - start) * 1000, 2)
    
    pred_idx = int(np.argmax(probs))
    pred_class_en = CLASS_NAMES[pred_idx]
    pred_class_es = CLASS_NAMES_ES.get(pred_class_en, pred_class_en)
    confidence = float(probs[pred_idx])

    prob_dict = {CLASS_NAMES_ES.get(CLASS_NAMES[i], CLASS_NAMES[i]): float(probs[i]) for i in range(len(CLASS_NAMES))}
    ranked = sorted(
        [
            {
                "className": CLASS_NAMES[i],
                "classNameEs": CLASS_NAMES_ES.get(CLASS_NAMES[i], CLASS_NAMES[i]),
                "probability": round(float(probs[i]) * 100, 2)
            }
            for i in range(len(CLASS_NAMES))
        ],
        key=lambda x: x["probability"],
        reverse=True
    )

    return {
        "predicted_class": pred_class_en,
        "predicted_class_es": pred_class_es,
        "confidence": round(confidence * 100, 2),
        "probabilities": prob_dict,
        "ranked_probabilities": ranked,
        "inference_time_ms": latency,
        "model_version": "Baseline-LogReg (Scikit-Learn)"
    }
