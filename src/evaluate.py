"""Evaluation script for chest X-ray classification model on test set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_dataloaders
from src.model import build_model
from src.utils import load_config


def _plot_confusion_matrix(all_labels, all_preds, class_names, output_path: Path) -> None:
    """Create and save confusion matrix heatmap."""
    cm = confusion_matrix(all_labels, all_preds)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to: {output_path}")


def _plot_roc_curve(all_labels, all_probs_pneumonia, auc, output_path: Path) -> None:
    """Create and save ROC curve."""
    fpr, tpr, _ = roc_curve(all_labels, all_probs_pneumonia)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved ROC curve to: {output_path}")


def _plot_pr_curve(all_labels, all_probs_pneumonia, output_path: Path) -> None:
    """Create and save Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(all_labels, all_probs_pneumonia)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label="PR Curve", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved PR curve to: {output_path}")


def _plot_per_class_metrics(
    per_class_precision, per_class_recall, per_class_f1, class_names, output_path: Path
) -> None:
    """Create and save per-class metrics bar chart."""
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, per_class_precision, width, label="Precision")
    ax.bar(x, per_class_recall, width, label="Recall")
    ax.bar(x + width, per_class_f1, width, label="F1")
    
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim([0, 1.1])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved per-class metrics to: {output_path}")


def main() -> None:
    """Evaluate model on test set and save metrics and plots."""
    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    data_cfg = config["data"]
    model_cfg = config["model"]
    paths_cfg = config["paths"]
    
    checkpoint_dir = PROJECT_ROOT / paths_cfg["checkpoint_dir"]
    plots_dir = PROJECT_ROOT / paths_cfg["plots_dir"]
    results_dir = PROJECT_ROOT / paths_cfg["results_dir"]
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    best_checkpoint_path = checkpoint_dir / "best_model.pth"
    if not best_checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {best_checkpoint_path}")
        return
    
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    epoch_trained = int(checkpoint["epoch"])
    best_val_auc = float(checkpoint.get("val_auc", 0.0))
    print(f"Loaded model from epoch {epoch_trained} with val AUC: {best_val_auc:.4f}")
    
    # Build and load model
    model = build_model(
        architecture=model_cfg["architecture"],
        num_classes=data_cfg["num_classes"],
        pretrained=False,
        dropout=model_cfg["dropout"],
        freeze_backbone=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Load test data
    _, _, test_loader, _ = get_dataloaders(
        data_dir=data_cfg["data_dir"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
    )
    
    # Inference loop
    all_labels = []
    all_preds = []
    all_probs_pneumonia = []
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs_pneumonia.extend(probs[:, 1].cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs_pneumonia = np.array(all_probs_pneumonia)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")
    auc = roc_auc_score(all_labels, all_probs_pneumonia)
    
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Per-class metrics
    per_class_precision = precision_score(all_labels, all_preds, average=None)
    per_class_recall = recall_score(all_labels, all_preds, average=None)
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    
    class_names = data_cfg["class_names"]
    
    # Print metrics table
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION METRICS")
    print("=" * 60)
    for metric_name, metric_value in [
        ("Accuracy", accuracy),
        ("Precision (weighted)", precision),
        ("Recall (weighted)", recall),
        ("F1 Score (weighted)", f1),
        ("AUC-ROC", auc),
        ("Sensitivity (recall PNEUMONIA)", sensitivity),
        ("Specificity (recall NORMAL)", specificity),
    ]:
        print(f"{metric_name:<35} {metric_value:.4f}")
    print("=" * 60)
    
    # Save plots
    _plot_confusion_matrix(
        all_labels,
        all_preds,
        class_names,
        plots_dir / "confusion_matrix.png",
    )
    _plot_roc_curve(all_labels, all_probs_pneumonia, auc, plots_dir / "roc_curve.png")
    _plot_pr_curve(all_labels, all_probs_pneumonia, plots_dir / "pr_curve.png")
    _plot_per_class_metrics(
        per_class_precision,
        per_class_recall,
        per_class_f1,
        class_names,
        plots_dir / "per_class_metrics.png",
    )
    
    # Save metrics to JSON
    metrics_dict = {
        "checkpoint_epoch": epoch_trained,
        "checkpoint_val_auc": best_val_auc,
        "test_accuracy": float(accuracy),
        "test_precision_weighted": float(precision),
        "test_recall_weighted": float(recall),
        "test_f1_weighted": float(f1),
        "test_auc_roc": float(auc),
        "test_sensitivity": float(sensitivity),
        "test_specificity": float(specificity),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "per_class": {
            class_names[i]: {
                "precision": float(per_class_precision[i]),
                "recall": float(per_class_recall[i]),
                "f1": float(per_class_f1[i]),
            }
            for i in range(len(class_names))
        },
    }
    
    metrics_path = results_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics_dict, file, indent=2)
    
    print(f"\nSaved test metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
