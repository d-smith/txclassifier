"""Evaluates the currently trained model (the same joblib artifacts train.py logs to
MLflow) against a labeled CSV, and logs the result as an MLflow run in a separate
"txclassifier-eval" experiment, so efficacy can be tracked over time independent of
training runs.

Runnable as a script: `uv run python -m src.evaluate_labeled`.
Optionally pass a CSV path to evaluate on data other than config.TRAINING_DATA_PATH,
e.g. `uv run python -m src.evaluate_labeled data/holdout/holdout_transactions.csv`.
"""

import argparse

import mlflow

from src import config, data_loader, evaluate, taxonomy
from src.predictor import SklearnPredictor


def run_eval(data_path) -> dict:
    df = data_loader.load_raw_transactions(data_path)
    clean_df, drop_report = data_loader.drop_unlabeled(df)
    evaluate.print_drop_report(drop_report)

    predictor = SklearnPredictor()

    rows = []
    for row in clean_df.itertuples(index=False):
        try:
            result = predictor.predict(row.description, row.amount, row.date)
        except ValueError:
            # No subcategory model available for the predicted category.
            rows.append(
                {
                    "true_category": row.category,
                    "pred_category": taxonomy.UNKNOWN_LABEL,
                    "true_subcategory": row.subcategory,
                    "pred_subcategory": None,
                    "confidence": None,
                }
            )
            continue
        rows.append(
            {
                "true_category": row.category,
                "pred_category": result.category,
                "true_subcategory": row.subcategory,
                "pred_subcategory": result.subcategory,
                "confidence": result.confidence,
            }
        )

    category_labels = [*taxonomy.CATEGORIES.keys(), taxonomy.UNKNOWN_LABEL]
    category_metrics = evaluate.evaluate_classifier(
        [r["true_category"] for r in rows],
        [r["pred_category"] for r in rows],
        category_labels,
        "category",
    )

    # Subcategory model choice depends on the *predicted* category, so subcategory
    # accuracy is only meaningful conditioned on a correct category prediction.
    category_correct = [r for r in rows if r["pred_category"] == r["true_category"]]
    subcategory_accuracy = (
        sum(r["pred_subcategory"] == r["true_subcategory"] for r in category_correct) / len(category_correct)
        if category_correct
        else 0.0
    )
    overall_accuracy = sum(
        r["pred_category"] == r["true_category"] and r["pred_subcategory"] == r["true_subcategory"] for r in rows
    ) / len(rows)
    confidences = [r["confidence"] for r in rows if r["confidence"] is not None]
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "num_rows": len(rows),
        "category_metrics": category_metrics,
        "subcategory_accuracy": subcategory_accuracy,
        "overall_accuracy": overall_accuracy,
        "mean_confidence": mean_confidence,
    }


def main(data_path=None) -> None:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("txclassifier-eval")

    eval_data_path = data_path or config.TRAINING_DATA_PATH
    results = run_eval(eval_data_path)

    evaluate.print_evaluation(results["category_metrics"])
    print(f"subcategory_accuracy (given correct category): {results['subcategory_accuracy']:.3f}")
    print(f"overall_accuracy (category AND subcategory correct): {results['overall_accuracy']:.3f}")
    print(f"mean_confidence: {results['mean_confidence']:.3f}")

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.METRICS_DIR / "eval_category_report.json"
    evaluate.save_evaluation(results["category_metrics"], report_path)

    with mlflow.start_run(run_name="labeled-eval"):
        mlflow.log_param("eval_data_path", str(eval_data_path))
        mlflow.log_metrics(
            {
                "num_rows": results["num_rows"],
                "category_accuracy": results["category_metrics"]["classification_report"]["accuracy"],
                "subcategory_accuracy": results["subcategory_accuracy"],
                "overall_accuracy": results["overall_accuracy"],
                "mean_confidence": results["mean_confidence"],
            }
        )
        mlflow.log_artifact(str(report_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path",
        nargs="?",
        default=None,
        help="Labeled CSV to evaluate on (date,description,amount,category,subcategory). "
        "Defaults to config.TRAINING_DATA_PATH.",
    )
    args = parser.parse_args()
    main(args.data_path)
