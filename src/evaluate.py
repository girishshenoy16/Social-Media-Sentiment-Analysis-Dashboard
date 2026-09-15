"""Test set evaluation and confusion matrix generator."""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "docs" / "outputs"


def compute_metrics(y_true, y_pred, class_map):
    """Compute and return all required metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
    }
    return metrics


def generate_confusion_matrix(y_true, y_pred, class_map, model_name):
    """Generate and save a confusion matrix plot."""
    labels = sorted(class_map.keys())
    label_names = [class_map[l] for l in labels]
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    cm = confusion_matrix(
        y_true_arr, y_pred_arr, labels=[int(l) for l in labels]
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=label_names, yticklabels=label_names,
        ax=ax,
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14)
    plt.tight_layout()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(OUTPUTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return OUTPUTS_DIR / filename


def save_metrics(metrics, filename="metrics_summary.json"):
    """Save metrics to a JSON file."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return path


def evaluate_model(model, X_test, y_test, class_map, model_name):
    """Evaluate a model on the test set and generate all outputs."""
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred, class_map)
    report = classification_report(
        y_test, y_pred,
        target_names=[class_map[i] for i in sorted(class_map.keys())],
        zero_division=0,
    )

    cm_path = generate_confusion_matrix(y_test, y_pred, class_map, model_name)

    print(f"\n--- {model_name} ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
    print(f"\n{report}")
    print(f"Confusion matrix saved to: {cm_path}")

    return metrics, y_pred
