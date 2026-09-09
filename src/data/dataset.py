import json
import os
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, SEED

def make_stratified_splits():
    dataset = ImageFolder(root=DATA_RAW_DIR)
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
        "train": train_idx,
        "dev": dev_idx,
        "test": test_idx,
        "classes": dataset.classes
    }
    
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    with open(os.path.join(DATA_PROCESSED_DIR, "splits.json"), "w") as f:
        json.dump(splits, f, indent=2)
        
    print(f"Splits generados: Train={len(train_idx)}, Dev={len(dev_idx)}, Test={len(test_idx)}")