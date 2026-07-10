"""Generic evaluation helpers shared by the category model and all 14 subcategory models."""

import json

from sklearn.metrics import classification_report, confusion_matrix

from src.data_loader import DropReport


def evaluate_classifier(y_true, y_pred, label_names, model_name: str) -> dict:
    """Computes classification_report + confusion_matrix for one model's predictions."""
    report = classification_report(
        y_true, y_pred, labels=label_names, target_names=label_names, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=label_names)
    return {
        "model_name": model_name,
        "labels": list(label_names),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def print_evaluation(metrics: dict) -> None:
    report = metrics["classification_report"]
    print(f"--- {metrics['model_name']} ---")
    print(f"accuracy: {report['accuracy']:.3f}")
    for label in metrics["labels"]:
        stats = report.get(label, {})
        print(
            f"  {label}: precision={stats.get('precision', 0):.2f} "
            f"recall={stats.get('recall', 0):.2f} f1={stats.get('f1-score', 0):.2f} "
            f"support={stats.get('support', 0):.0f}"
        )


def save_evaluation(metrics: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def print_drop_report(report: DropReport) -> None:
    print(report)
