"""The single shared preprocessing pipeline used identically by training and serving.

build_preprocessor() is fit exactly once (in train.py) and persisted as
preprocessor.joblib. Every other consumer (the 14 subcategory training loops,
evaluate.py, and SklearnPredictor at serve time) loads that fitted object and
only ever calls .transform() on it -- this is what rules out train/serve skew
by construction rather than by convention.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DayOfWeekExtractor(BaseEstimator, TransformerMixin):
    """Parses the `date` column into a single day-of-week integer column (0=Monday..6=Sunday)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dates = pd.to_datetime(X["date"], errors="coerce")
        day_of_week = dates.dt.dayofweek.fillna(-1).astype(int).to_numpy().reshape(-1, 1)
        return day_of_week

    def get_feature_names_out(self, input_features=None):
        return np.array(["day_of_week"])


def build_preprocessor() -> ColumnTransformer:
    """Returns the shared ColumnTransformer: TF-IDF char n-grams on description,
    imputed+scaled amount, and one-hot-encoded day-of-week derived from date.
    """
    amount_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    day_of_week_pipeline = Pipeline(
        steps=[
            ("extract", DayOfWeekExtractor()),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("description", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)), "description"),
            ("amount", amount_pipeline, ["amount"]),
            ("day_of_week", day_of_week_pipeline, ["date"]),
        ]
    )
