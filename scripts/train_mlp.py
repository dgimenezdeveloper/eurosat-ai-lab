import os
import sys
import time
import json
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import DATA_RAW_DIR, SPLITS_PATH, ARTIFACTS_DIR, SEED, DEVICE
from src.models.mlp import EurosatMLP
from src.training.tracker import log_experiment

def main():
    print("=== [1/4] Cargando datos y escalador ===")
    baseline_path = os.path.join(ARTIFACTS_DIR, "models", "baseline_logreg.joblib")
    if not os.path.exists(baseline_path):
        print("[ERROR] Falta baseline_logreg.joblib. Ejecutá train_baseline.py primero.")
        return
    artifact = joblib.load(baseline_path)
    scaler = artifact["scaler"]

    with open(SPLITS_PATH, "r") as f:
        splits = json.load(f)

    dataset = ImageFolder(root=DATA_RAW_DIR)

    def get_data(indices, max_n):
        X, y = [], []
        for idx in indices[:max_n]:
            img, target = dataset[idx]
            X.append(np.array(img, dtype=np.float32).flatten())
            y.append(target)
        return scaler.transform(np.array(X)), np.array(y)

    print("Extrayendo matrices de píxeles...")
    X_train, y_train = get_data(splits["train_indices"], max_n=4000)
    X_dev, y_dev = get_data(splits["dev_indices"], max_n=1000)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=128, shuffle=True
    )
    dev_loader = DataLoader(
        TensorDataset(torch.tensor(X_dev, dtype=torch.float32), torch.tensor(y_dev, dtype=torch.long)),
        batch_size=128, shuffle=False
    )

    print("\n=== [2/4] Entrenando MLP Regularizado y Registrando Curvas de Aprendizaje ===")
    torch.manual_seed(SEED)
    model = EurosatMLP(input_dim=12288, num_classes=10, regularized=True, dropout_rate=0.35)
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    history = {
        "epochs": [],
        "train_loss": [],
        "dev_loss": [],
        "train_acc": [],
        "dev_acc": []
    }

    start_t = time.time()
    num_epochs = 12

    for epoch in range(1, num_epochs + 1):
        # Fase de entrenamiento
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        total_train = 0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * X_b.size(0)
            preds = torch.argmax(out, dim=1)
            train_correct += (preds == y_b).sum().item()
            total_train += y_b.size(0)

        epoch_train_loss = running_train_loss / total_train
        epoch_train_acc = (train_correct / total_train) * 100

        # Fase de validación por época
        model.eval()
        running_dev_loss = 0.0
        dev_correct = 0
        total_dev = 0

        with torch.no_grad():
            for X_b, y_b in dev_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                out = model(X_b)
                loss = criterion(out, y_b)
                running_dev_loss += loss.item() * X_b.size(0)
                preds = torch.argmax(out, dim=1)
                dev_correct += (preds == y_b).sum().item()
                total_dev += y_b.size(0)

        epoch_dev_loss = running_dev_loss / total_dev
        epoch_dev_acc = (dev_correct / total_dev) * 100

        history["epochs"].append(epoch)
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["dev_loss"].append(round(epoch_dev_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 2))
        history["dev_acc"].append(round(epoch_dev_acc, 2))

        print(f" -> Época {epoch:02d}/{num_epochs:02d} | Train Loss: {epoch_train_loss:.4f} | Dev Acc: {epoch_dev_acc:.2f}%")

    train_time = time.time() - start_t
    print(f"✓ Entrenamiento finalizado en {train_time:.1f} s.")

    print("\n=== [3/4] Evaluando Métricas y Generando Matriz de Confusión ===")
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_b, y_b in dev_loader:
            logits = model(X_b.to(DEVICE))
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_b.numpy())

    dev_acc = accuracy_score(all_targets, all_preds) * 100
    dev_f1 = f1_score(all_targets, all_preds, average="macro")
    latency = (train_time / len(X_dev)) * 1000.0

    cm = confusion_matrix(all_targets, all_preds).tolist()

    print(f" • Exactitud Final en Dev: {dev_acc:.2f}%")
    print(f" • Macro F1-Score:         {dev_f1:.4f}")

    print("\n=== [4/4] Guardando Artefactos y Ledger ===")
    models_dir = os.path.join(ARTIFACTS_DIR, "models")
    metrics_dir = os.path.join(ARTIFACTS_DIR, "metrics")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # 1. Checkpoint
    torch.save(model.state_dict(), os.path.join(models_dir, "mlp_regularized.pt"))
    # 2. Curvas de aprendizaje
    with open(os.path.join(metrics_dir, "mlp_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    # 3. Matriz de confusión
    with open(os.path.join(metrics_dir, "confusion_matrix_mlp.json"), "w") as f:
        json.dump(cm, f)

    log_experiment(
        model_name="MLP Regularizado (BatchNorm+Drop+L2)",
        hyperparameters={"hidden_layers": [512, 256, 128], "activation": "GELU", "dropout": 0.35, "lr": 1e-3, "weight_decay": 1e-3},
        metrics={
            "train_accuracy": round(history["train_acc"][-1], 2),
            "dev_accuracy": round(dev_acc, 2),
            "macro_f1": round(dev_f1, 4),
            "latency_ms": round(latency, 2)
        },
        dataset_info={"train_samples": 4000, "dev_samples": 1000},
        notes="Etapa 2: Curvas de aprendizaje por época y matriz de confusión serializadas."
    )
    print("\n✓ ¡Artefactos exportados! Listo para recargar Streamlit.")

if __name__ == "__main__":
    main()