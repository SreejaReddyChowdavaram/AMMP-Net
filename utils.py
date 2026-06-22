# -*- coding: utf-8 -*-
# utils.py
# Seeding, parameter counting, file saving, and validation helper utilities for AMMP-Net.

import os
import random
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# -------------------------------------------------------------------------
# Seeding for reproducibility
# -------------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    """
    Sets seed for all packages to ensure deterministic run outcomes.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# -------------------------------------------------------------------------
# Parameter count helper
# -------------------------------------------------------------------------
def parameter_counter(model: torch.nn.Module) -> int:
    """
    Returns the count of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------------------------------------------------------
# Per-class accuracy calculation
# -------------------------------------------------------------------------
def compute_class_accuracy(cm: np.ndarray) -> np.ndarray:
    """
    Computes per-class classification accuracy percentages from a confusion matrix.
    """
    per_class_acc = []
    num_classes = cm.shape[0]
    for i in range(num_classes):
        row_sum = cm[i].sum()
        if row_sum > 0:
            acc = float(cm[i, i]) / float(row_sum) * 100.0
        else:
            acc = 0.0
        per_class_acc.append(acc)
    return np.array(per_class_acc, dtype=np.float32)


# -------------------------------------------------------------------------
# Save Configuration to File
# -------------------------------------------------------------------------
def save_config(config: dict, run_dir: str) -> None:
    """
    Saves the experiment configuration settings to config.txt.
    """
    path = os.path.join(run_dir, "config.txt")
    with open(path, "w", encoding="utf-8") as f:
        for k, v in sorted(config.items()):
            f.write(f"{k}: {v}\n")
    print(f"[Logging] Saved configuration settings to {path}")


# -------------------------------------------------------------------------
# Save Predictions to File
# -------------------------------------------------------------------------
def save_predictions(predictions: np.ndarray, run_dir: str) -> None:
    """
    Saves the final test predictions to best_predictions.npz.
    """
    path = os.path.join(run_dir, "best_predictions.npz")
    np.savez_compressed(path, predictions=predictions)
    print(f"[Logging] Saved model predictions to {path}")


# -------------------------------------------------------------------------
# Save Metrics to File
# -------------------------------------------------------------------------
def save_metrics(metrics: dict, run_dir: str, dataset_name: str, cfg: dict, dinfo: dict) -> None:
    """
    Saves the final experimental metrics in the specified publication-ready text format.
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join(run_dir, "metrics.txt")
    
    oa = metrics.get("oa", 0.0)
    aa = metrics.get("aa", 0.0)
    kappa = metrics.get("kappa", 0.0)
    
    best_epoch = metrics.get("best_epoch", 0)
    n_params = metrics.get("model_parameters", 0)
    train_samples = dinfo.get("num_source_train", 0)
    test_samples = dinfo.get("num_target_test", 0)
    
    lr = cfg.get("learning_rate", 0.0)
    batch_size = cfg.get("batch_size", 0)
    epochs = cfg.get("epochs", 0)
    
    cm = metrics.get("confusion_matrix", np.zeros((1, 1)))
    per_class_acc = metrics.get("per_class_accuracy", [])
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Dataset\n{dataset_name}\n\n")
        f.write(f"Timestamp\n{timestamp}\n\n")
        f.write(f"Best Epoch\n{best_epoch}\n\n")
        f.write(f"OA\n{oa:.4f}%\n\n")
        f.write(f"AA\n{aa:.4f}%\n\n")
        f.write(f"Kappa\n{kappa*100:.4f}%\n\n")
        f.write(f"Model Parameters\n{n_params}\n\n")
        f.write(f"Training Samples\n{train_samples}\n\n")
        f.write(f"Testing Samples\n{test_samples}\n\n")
        f.write(f"Learning Rate\n{lr}\n\n")
        f.write(f"Batch Size\n{batch_size}\n\n")
        f.write(f"Epochs\n{epochs}\n\n")
        
        f.write("==================================================\n")
        f.write("Per-Class Accuracy\n")
        f.write("==================================================\n")
        for i, acc in enumerate(per_class_acc):
            f.write(f"Class {i+1}: {acc:.4f}%\n")
        f.write("\n")
        
        f.write("==================================================\n")
        f.write("Confusion Matrix\n")
        f.write("==================================================\n")
        n_classes = cm.shape[0]
        for i in range(n_classes):
            row_str = " ".join(str(int(val)) for val in cm[i])
            f.write(f"{row_str}\n")
        f.write("==================================================\n")
        
    print(f"[Logging] Saved publication metrics report to {path}")


# -------------------------------------------------------------------------
# Model evaluation function (Merged from evaluation.py)
# -------------------------------------------------------------------------
def evaluate_model(model: torch.nn.Module, loader: torch.utils.data.DataLoader,
                   device: torch.device) -> dict:
    """
    Evaluates the model on the provided data loader and computes performance metrics.
    Returns:
        A dictionary containing 'oa', 'aa', 'kappa', 'confusion_matrix',
        'per_class_accuracy', and 'predictions'.
    """
    model.eval()
    preds_all = []
    labels_all = []
    
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, _ = batch
            elif len(batch) == 2:
                x, y = batch
            else:
                x = batch
                y = None
                
            x = x.to(device)
            # Forward student backbone
            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            
            preds = logits.argmax(dim=1).cpu().numpy()
            preds_all.extend(preds.tolist())
            
            if y is not None:
                labels_all.extend(y.numpy().tolist())
                
    preds_all = np.array(preds_all, dtype=np.int32)
    
    if len(labels_all) == 0:
        return {
            "predictions": preds_all
        }
        
    labels_all = np.array(labels_all, dtype=np.int32)
    num_classes = max(preds_all.max(), labels_all.max()) + 1
    
    cm = confusion_matrix(labels_all, preds_all, labels=np.arange(num_classes))
    
    per_class_acc = []
    for i in range(num_classes):
        row_sum = cm[i].sum()
        if row_sum > 0:
            acc = float(cm[i, i]) / float(row_sum) * 100.0
        else:
            acc = 0.0
        per_class_acc.append(acc)
        
    oa = float(np.mean(preds_all == labels_all)) * 100.0
    aa = float(np.mean(per_class_acc))
    kappa = float(cohen_kappa_score(labels_all, preds_all))
    
    return {
        "oa": oa,
        "aa": aa,
        "kappa": kappa,
        "confusion_matrix": cm,
        "per_class_accuracy": per_class_acc,
        "predictions": preds_all,
        "labels": labels_all
    }

