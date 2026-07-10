"""Paths, hyperparameters, and random seed shared across the pipeline."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
SYNTHETIC_DATA_PATH = DATA_DIR / "synthetic" / "synthetic_transactions.csv"
RAW_DATA_PATH = DATA_DIR / "raw" / "transactions.csv"

MODELS_DIR = REPO_ROOT / "models"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
CATEGORY_MODEL_PATH = MODELS_DIR / "category_model.joblib"
CATEGORY_LABEL_ENCODER_PATH = MODELS_DIR / "category_label_encoder.joblib"
SUBCATEGORY_MODELS_DIR = MODELS_DIR / "subcategory_models"
METRICS_DIR = MODELS_DIR / "metrics"
METADATA_PATH = MODELS_DIR / "metadata.json"

RANDOM_SEED = 42
TEST_SIZE = 0.2

# Active data source for train.py; repointed to RAW_DATA_PATH once real data is available (M8).
TRAINING_DATA_PATH = SYNTHETIC_DATA_PATH
