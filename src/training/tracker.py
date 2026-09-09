import os
import json
import time
from datetime import datetime
from src.config import ARTIFACTS_DIR

LEDGER_PATH = os.path.join(ARTIFACTS_DIR, "metrics", "experiments_ledger.json")

def log_experiment(
    model_name: str,
    hyperparameters: dict,
    metrics: dict,
    dataset_info: dict,
    notes: str = ""
):
    """Guarda un registro inmutable de cada corrida/experimento de entrenamiento."""
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    
    ledger = []
    if os.path.exists(LEDGER_PATH):
        try:
            with open(LEDGER_PATH, "r") as f:
                ledger = json.load(f)
        except Exception:
            ledger = []

    run_number = len(ledger) + 1
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = f"EXP-{run_number:03d}"

    # Diagnóstico automático de Sesgo vs Varianza (Clase 3)
    train_acc = metrics.get("train_accuracy", 0.0)
    dev_acc = metrics.get("dev_accuracy", 0.0)
    gap = round(abs(train_acc - dev_acc), 2)
    
    if dev_acc < 40.0:
        diagnosis = "Sesgo Alto (Underfitting Estructural)"
    elif gap > 10.0:
        diagnosis = f"Varianza Alta (Overfitting, Gap={gap}%)"
    else:
        diagnosis = f"Equilibrado (Gap={gap}%)"

    entry = {
        "run_id": run_id,
        "timestamp": timestamp,
        "model_name": model_name,
        "hyperparameters": hyperparameters,
        "metrics": {
            "train_accuracy": train_acc,
            "dev_accuracy": dev_acc,
            "macro_f1": metrics.get("macro_f1", 0.0),
            "latency_ms": metrics.get("latency_ms", 0.0),
            "gap": gap
        },
        "dataset": dataset_info,
        "diagnosis": diagnosis,
        "notes": notes
    }

    # Guardar en el histórico general (append)
    ledger.insert(0, entry) # El más reciente primero
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2)

    # Actualizar models_summary.json con el mejor o último modelo
    summary_path = os.path.join(ARTIFACTS_DIR, "metrics", "models_summary.json")
    summary_entry = [
        {
            "name": f"{model_name} ({run_id})",
            "version": f"C={hyperparameters.get('C', 1.0)} | iter={hyperparameters.get('max_iter', 100)}",
            "accuracy": dev_acc,
            "f1": metrics.get("macro_f1", 0.0),
            "latency": metrics.get("latency_ms", 0.0),
            "status": "Evaluated"
        }
    ]
    with open(summary_path, "w") as f:
        json.dump(summary_entry, f, indent=2)

    print(f"\n[MLOps Tracker] Experimento registrado con éxito: {run_id}")
    print(f" -> Diagnóstico: {diagnosis}")
    return entry
