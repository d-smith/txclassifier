"""The model/serving boundary: serve/main.py depends only on the Predictor Protocol,
never on SklearnPredictor's internals, so a future TransformerPredictor can swap in
with a one-line change and no endpoint/schema changes.
"""

from dataclasses import dataclass
from typing import Protocol

import joblib
import pandas as pd

from src import config


@dataclass
class PredictionResult:
    category: str
    subcategory: str
    confidence: float
    category_confidence: float
    subcategory_confidence: float


class Predictor(Protocol):
    def predict(self, description: str, amount: float, date: str) -> PredictionResult: ...

    def health_check(self) -> bool: ...


class SklearnPredictor:
    """Loads the shared preprocessor + category model + all subcategory models once at startup."""

    def __init__(self):
        self.preprocessor = joblib.load(config.PREPROCESSOR_PATH)
        self.category_model = joblib.load(config.CATEGORY_MODEL_PATH)
        self.category_encoder = joblib.load(config.CATEGORY_LABEL_ENCODER_PATH)

        self.subcategory_models: dict[str, object] = {}
        self.subcategory_encoders: dict[str, object] = {}
        from src.taxonomy import CATEGORIES, slugify_category

        for category in CATEGORIES:
            slug = slugify_category(category)
            model_path = config.SUBCATEGORY_MODELS_DIR / f"{slug}_model.joblib"
            encoder_path = config.SUBCATEGORY_MODELS_DIR / f"{slug}_label_encoder.joblib"
            if model_path.exists() and encoder_path.exists():
                self.subcategory_models[category] = joblib.load(model_path)
                self.subcategory_encoders[category] = joblib.load(encoder_path)

    def predict(self, description: str, amount: float, date: str) -> PredictionResult:
        row = pd.DataFrame([{"description": description, "amount": amount, "date": date}])
        X = self.preprocessor.transform(row)

        category_probs = self.category_model.predict_proba(X)[0]
        category_idx = category_probs.argmax()
        category = self.category_encoder.inverse_transform([category_idx])[0]
        category_confidence = float(category_probs[category_idx])

        if category not in self.subcategory_models:
            raise ValueError(f"No subcategory model available for category '{category}'")

        subcategory_model = self.subcategory_models[category]
        subcategory_encoder = self.subcategory_encoders[category]
        subcategory_probs = subcategory_model.predict_proba(X)[0]
        subcategory_idx = subcategory_probs.argmax()
        subcategory = subcategory_encoder.inverse_transform([subcategory_idx])[0]
        subcategory_confidence = float(subcategory_probs[subcategory_idx])

        return PredictionResult(
            category=category,
            subcategory=subcategory,
            confidence=category_confidence * subcategory_confidence,
            category_confidence=category_confidence,
            subcategory_confidence=subcategory_confidence,
        )

    def health_check(self) -> bool:
        return (
            self.preprocessor is not None
            and self.category_model is not None
            and len(self.subcategory_models) > 0
        )
