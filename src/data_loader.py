"""CSV loading and label-cleanliness handling.

drop_unlabeled() is the *only* thing that touches label cleanliness for
training purposes: rows with a missing category or subcategory are dropped
entirely and never seen by .fit(). label_or_unknown() is a separate,
reporting-only helper -- "Unknown" is a display label for humans, never a
class the model learns to predict.
"""

from dataclasses import dataclass, field

import pandas as pd

from src.taxonomy import UNKNOWN_LABEL


def _is_missing(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "")


def load_raw_transactions(path) -> pd.DataFrame:
    """Read a transactions CSV untouched (date, description, amount, category, subcategory)."""
    return pd.read_csv(path)


@dataclass
class DropReport:
    total_rows: int
    kept_rows: int
    dropped_rows: int
    dropped_by_category: dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"Total rows: {self.total_rows}",
            f"Kept rows: {self.kept_rows}",
            f"Dropped rows (missing category/subcategory): {self.dropped_rows}",
        ]
        if self.dropped_by_category:
            lines.append("Dropped by category:")
            for category, count in sorted(self.dropped_by_category.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {category}: {count}")
        return "\n".join(lines)


def drop_unlabeled(df: pd.DataFrame) -> tuple[pd.DataFrame, DropReport]:
    """Drop rows with a missing category or subcategory. Never keeps 'Unknown' as a trainable class.

    Returns (clean_df, drop_report) where drop_report breaks the dropped-row
    count down by category (rows with a missing category itself are bucketed
    under "Unknown").
    """
    missing_mask = _is_missing(df["category"]) | _is_missing(df["subcategory"])

    dropped = df[missing_mask]
    clean_df = df[~missing_mask].reset_index(drop=True)

    dropped_by_category: dict[str, int] = {}
    for category in dropped["category"]:
        key = category if isinstance(category, str) and category.strip() else UNKNOWN_LABEL
        dropped_by_category[key] = dropped_by_category.get(key, 0) + 1

    report = DropReport(
        total_rows=len(df),
        kept_rows=len(clean_df),
        dropped_rows=len(dropped),
        dropped_by_category=dropped_by_category,
    )
    return clean_df, report


def label_or_unknown(value) -> str:
    """Reporting/display-only helper: blank/null -> 'Unknown'. Never feeds .fit()."""
    if not isinstance(value, str) or not value.strip():
        return UNKNOWN_LABEL
    return value
