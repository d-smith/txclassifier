"""Trains the category model + 14 subcategory models, saving artifacts and metrics.

Flow: load -> drop_unlabeled (print report) -> stratified 80/20 split -> fit
preprocessor once, dump it -> train category LogisticRegression + LabelEncoder
-> loop over taxonomy.CATEGORIES (not over data, so an empty category is
reported as skipped rather than silently missing), transform each category's
subset with the already-fit preprocessor, train that category's subcategory
LogisticRegression, save model+encoder+metrics -> dump metadata.json.

Runnable as a script: `uv run python -m src.train`.
Optionally pass a CSV path to train on data other than config.TRAINING_DATA_PATH,
e.g. `uv run python -m src.train data/combined/combined_transactions.csv`.
"""

import argparse
import json
import warnings
from datetime import datetime, timezone

import joblib
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src import config, data_loader, evaluate, features, taxonomy
from src.taxonomy import slugify_category


def _stratified_split(df):
    """Splits on the combined category|subcategory key; falls back to an
    unstratified split with a warning if any combined class has <2 members.
    """
    combined_key = df["category"] + "|" + df["subcategory"]
    class_counts = combined_key.value_counts()
    if (class_counts < 2).any():
        warnings.warn(
            f"{(class_counts < 2).sum()} category|subcategory combination(s) have fewer than 2 rows; "
            "falling back to a non-stratified split.",
            stacklevel=2,
        )
        return train_test_split(df, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)
    return train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=combined_key
    )


def train_category_model(preprocessor, train_df, test_df):
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df["category"])
    y_test = encoder.transform(test_df["category"])

    assert set(encoder.classes_) <= set(taxonomy.CATEGORIES.keys()), (
        "Category model learned classes outside the taxonomy"
    )

    model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate.evaluate_classifier(
        encoder.inverse_transform(y_test), encoder.inverse_transform(y_pred), list(encoder.classes_), "category"
    )
    return model, encoder, metrics


def train_subcategory_models(preprocessor, train_df, test_df):
    """Loops over taxonomy.CATEGORIES (not over data) so an empty category is
    reported as skipped, not silently missing.
    """
    results = {}
    for category in taxonomy.CATEGORIES:
        slug = slugify_category(category)
        cat_train = train_df[train_df["category"] == category]
        cat_test = test_df[test_df["category"] == category]

        if len(cat_train) == 0 or cat_train["subcategory"].nunique() < 2:
            print(f"  [{category}] skipped (insufficient training data)")
            results[category] = None
            continue

        X_train = preprocessor.transform(cat_train)
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(cat_train["subcategory"])

        assert set(encoder.classes_) <= set(taxonomy.CATEGORIES[category]), (
            f"Subcategory model for '{category}' learned classes outside the taxonomy"
        )

        model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X_train, y_train)

        metrics = None
        if len(cat_test) > 0:
            X_test = preprocessor.transform(cat_test)
            y_test = encoder.transform(cat_test["subcategory"])
            y_pred = model.predict(X_test)
            metrics = evaluate.evaluate_classifier(
                encoder.inverse_transform(y_test),
                encoder.inverse_transform(y_pred),
                list(encoder.classes_),
                f"subcategory_{slug}",
            )

        results[category] = (model, encoder, metrics, slug)

    return results


def main(data_path=None) -> None:
    training_data_path = data_path or config.TRAINING_DATA_PATH
    df = data_loader.load_raw_transactions(training_data_path)
    clean_df, drop_report = data_loader.drop_unlabeled(df)
    evaluate.print_drop_report(drop_report)

    train_df, test_df = _stratified_split(clean_df)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.SUBCATEGORY_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    preprocessor = features.build_preprocessor()
    category_model, category_encoder, category_metrics = train_category_model(preprocessor, train_df, test_df)
    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)
    joblib.dump(category_model, config.CATEGORY_MODEL_PATH)
    joblib.dump(category_encoder, config.CATEGORY_LABEL_ENCODER_PATH)
    evaluate.print_evaluation(category_metrics)
    evaluate.save_evaluation(category_metrics, config.METRICS_DIR / "category.json")

    print("\nTraining subcategory models:")
    subcategory_results = train_subcategory_models(preprocessor, train_df, test_df)

    skipped = []
    for category, result in subcategory_results.items():
        slug = slugify_category(category)
        if result is None:
            skipped.append(category)
            continue
        model, encoder, metrics, slug = result
        joblib.dump(model, config.SUBCATEGORY_MODELS_DIR / f"{slug}_model.joblib")
        joblib.dump(encoder, config.SUBCATEGORY_MODELS_DIR / f"{slug}_label_encoder.joblib")
        if metrics is not None:
            evaluate.print_evaluation(metrics)
            evaluate.save_evaluation(metrics, config.METRICS_DIR / f"subcategory_{slug}.json")

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "training_data_path": str(training_data_path),
        "total_rows": drop_report.total_rows,
        "dropped_rows": drop_report.dropped_rows,
        "kept_rows": drop_report.kept_rows,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "taxonomy_categories": len(taxonomy.CATEGORIES),
        "taxonomy_subcategories": sum(len(v) for v in taxonomy.CATEGORIES.values()),
        "skipped_categories": skipped,
    }
    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nWrote metadata to {config.METADATA_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path",
        nargs="?",
        default=None,
        help="CSV to train on (date,description,amount,category,subcategory). "
        "Defaults to config.TRAINING_DATA_PATH.",
    )
    args = parser.parse_args()
    main(args.data_path)
