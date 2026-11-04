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

# Asegurar resolución del módulo src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import DATA_RAW_DIR, SPLITS_PATH, ARTIFACTS_DIR, SEED, DEVICE
from src.models.mlp import EurosatMLP
from src.training.tracker import log_experiment


def run_training_cycle(model, optimizer, criterion, train_loader, dev_loader, epochs=12, name="Modelo"):
    """Ejecuta el ciclo de entrenamiento y validación por época registrando la telemetría clínica."""
    print(f"\n--- Iniciando entrenamiento: {name} ({epochs} Épocas) ---")
    history = {
        "epochs": [],
        "train_loss": [],
        "dev_loss": [],
        "train_acc": [],
        "dev_acc": []
    }
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        # 1. Fase de Entrenamiento
        model.train()
        running_train_loss, train_correct, total_train = 0.0, 0, 0
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
        epoch_train_acc = (train_correct / total_train) * 100.0

        # 2. Fase de Validación (Dev Set)
        model.eval()
        running_dev_loss, dev_correct, total_dev = 0.0, 0, 0
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
        epoch_dev_acc = (dev_correct / total_dev) * 100.0

        history["epochs"].append(epoch)
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["dev_loss"].append(round(epoch_dev_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 2))
        history["dev_acc"].append(round(epoch_dev_acc, 2))

        print(
            f"  • Época {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} | Dev Loss: {epoch_dev_loss:.4f} | "
            f"Train Acc: {epoch_train_acc:5.2f}% | Dev Acc: {epoch_dev_acc:5.2f}%"
        )

    elapsed_time = time.time() - start_time
    print(f"✓ {name} finalizado en {elapsed_time:.1f} s.")
    return history, elapsed_time


def main():
    print("=" * 80)
    print("ETAPA 2: ENTRENAMIENTO COMPARATIVO MLP (VANILLA vs. REGULARIZADO)")
    print("Requisitos de Consigna 4.2.2 y 4.2.3 — Cátedra Lic. Pablo Moreira")
    print("=" * 80)

    # 1. Carga de datos y escalador del Baseline
    print("\n=== [1/4] Cargando splits y escalador estándar ===")
    baseline_path = os.path.join(ARTIFACTS_DIR, "models", "baseline_logreg.joblib")
    if not os.path.exists(baseline_path):
        print("[ERROR] Falta baseline_logreg.joblib. Ejecutá primero: python scripts/train_baseline.py")
        return
    artifact = joblib.load(baseline_path)
    scaler = artifact["scaler"]

    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    dataset = ImageFolder(root=DATA_RAW_DIR)

    def extract_and_scale(indices, max_n):
        X, y = [], []
        for idx in indices[:max_n]:
            img, target = dataset[idx]
            X.append(np.array(img, dtype=np.float32).flatten())
            y.append(target)
        return scaler.transform(np.array(X)), np.array(y)

    print("Extrayendo matrices de píxeles aplanadas (Train: 4000, Dev: 1000)...")
    X_train, y_train = extract_and_scale(splits["train_indices"], max_n=4000)
    X_dev, y_dev = extract_and_scale(splits["dev_indices"], max_n=1000)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=128, shuffle=True
    )
    dev_loader = DataLoader(
        TensorDataset(torch.tensor(X_dev, dtype=torch.float32), torch.tensor(y_dev, dtype=torch.long)),
        batch_size=128, shuffle=False
    )

    # 2. Experimento A: MLP Vanilla (Sin Regularización)
    # Permite evidenciar el sobreajuste severo requerido por la consigna 4.2.3
    torch.manual_seed(SEED)
    vanilla_model = EurosatMLP(input_dim=12288, num_classes=10, regularized=False)
    vanilla_model.to(DEVICE)
    vanilla_criterion = nn.CrossEntropyLoss()
    vanilla_opt = optim.AdamW(vanilla_model.parameters(), lr=1e-3, weight_decay=0.0)

    vanilla_history, vanilla_time = run_training_cycle(
        vanilla_model, vanilla_opt, vanilla_criterion, train_loader, dev_loader,
        epochs=12, name="1. MLP Vanilla (Sin Regularizar - Baseline Densa)"
    )

    # 3. Experimento B: MLP Regularizado (BatchNorm + Dropout(0.35) + L2 Weight Decay)
    torch.manual_seed(SEED)
    reg_model = EurosatMLP(input_dim=12288, num_classes=10, regularized=True, dropout_rate=0.35)
    reg_model.to(DEVICE)
    reg_criterion = nn.CrossEntropyLoss()
    reg_opt = optim.AdamW(reg_model.parameters(), lr=1e-3, weight_decay=1e-3)

    reg_history, reg_time = run_training_cycle(
        reg_model, reg_opt, reg_criterion, train_loader, dev_loader,
        epochs=12, name="2. MLP Regularizado (BatchNorm + Dropout + L2)"
    )

    # 4. Evaluación en Dev y Métricas Finales
    print("\n=== [3/4] Evaluando Métricas y Generando Matriz de Confusión ===")
    reg_model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_b, y_b in dev_loader:
            logits = reg_model(X_b.to(DEVICE))
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_b.numpy())

    dev_acc = accuracy_score(all_targets, all_preds) * 100.0
    dev_f1 = f1_score(all_targets, all_preds, average="macro")
    latency = (reg_time / len(X_dev)) * 1000.0
    cm_mlp = confusion_matrix(all_targets, all_preds).tolist()

    # Diagnóstico formal del Gap
    gap_vanilla = vanilla_history["train_acc"][-1] - vanilla_history["dev_acc"][-1]
    gap_reg = reg_history["train_acc"][-1] - dev_acc

    print(f"\n" + "-" * 75)
    print("RESULTADOS CLÍNICOS DE LA REGULARIZACIÓN (Dev Set):")
    print(f" • MLP Vanilla      -> Train Acc: {vanilla_history['train_acc'][-1]:.2f}% | Dev Acc: {vanilla_history['dev_acc'][-1]:.2f}% | Brecha (Gap): {gap_vanilla:.2f}% (Alta Varianza)")
    print(f" • MLP Regularizado -> Train Acc: {reg_history['train_acc'][-1]:.2f}% | Dev Acc: {dev_acc:.2f}% | Brecha (Gap): {gap_reg:.2f}% (Controlada)")
    print(f" • Macro F1-Score Final (Regularizado): {dev_f1:.4f}")
    print("-" * 75)

    # 5. Exportación de Artefactos Oficiales
    print("\n=== [4/4] Serializando Artefactos y Actualizando Ledger MLOps ===")
    models_dir = os.path.join(ARTIFACTS_DIR, "models")
    metrics_dir = os.path.join(ARTIFACTS_DIR, "metrics")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Checkpoint oficial
    ckpt_path = os.path.join(models_dir, "mlp_regularized.pt")
    torch.save(reg_model.state_dict(), ckpt_path)
    print(f"✓ Checkpoint PyTorch guardado: {ckpt_path}")

    # Historial para Streamlit (mantiene compatibilidad directa)
    with open(os.path.join(metrics_dir, "mlp_history.json"), "w", encoding="utf-8") as f:
        json.dump(reg_history, f, indent=2)

    # Historial comparativo completo (para cumplir con el punto 4.2.3)
    comparison_payload = {
        "epochs": reg_history["epochs"],
        "vanilla": vanilla_history,
        "regularized": reg_history
    }
    with open(os.path.join(metrics_dir, "mlp_comparison_history.json"), "w", encoding="utf-8") as f:
        json.dump(comparison_payload, f, indent=2)
    print(f"✓ Curvas de aprendizaje comparativas guardadas en: {os.path.join(metrics_dir, 'mlp_comparison_history.json')}")

    # Matriz de confusión
    with open(os.path.join(metrics_dir, "confusion_matrix_mlp.json"), "w", encoding="utf-8") as f:
        json.dump(cm_mlp, f, indent=2)
    print("✓ Matriz de confusión serializada.")

    # Registrar experimento en el Ledger MLOps
    log_experiment(
        model_name="MLP Regularizado (BatchNorm+Drop+L2)",
        hyperparameters={
            "hidden_layers": [512, 256, 128],
            "activation": "GELU",
            "dropout": 0.35,
            "batch_norm": True,
            "lr": 1e-3,
            "weight_decay": 1e-3
        },
        metrics={
            "train_accuracy": round(reg_history["train_acc"][-1], 2),
            "dev_accuracy": round(dev_acc, 2),
            "macro_f1": round(dev_f1, 4),
            "latency_ms": round(latency, 2)
        },
        dataset_info={"train_samples": 4000, "dev_samples": 1000},
        notes="Etapa 2: Comparativa de regularización (Vanilla vs Regularizado) y matriz de confusión serializada."
    )

    print("\n✓ ¡Entrenamiento completado exitosamente! Recargá la consola de Streamlit.")


if __name__ == "__main__":
    main()