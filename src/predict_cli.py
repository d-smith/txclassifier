"""Tiny CLI for manually eyeballing predictions before wiring up FastAPI (Milestone 5 gate).

Usage:
  uv run python -m src.predict_cli                      # runs the built-in sanity-check examples
  uv run python -m src.predict_cli "STARBUCKS #4521" 5.75 2025-06-10
"""

import sys

from src.predictor import SklearnPredictor

# Hand-picked examples spanning several categories, for the Milestone 5 sanity gate.
SANITY_EXAMPLES = [
    ("STARBUCKS #4521", 5.75, "2025-06-10"),
    ("TRADER JOE'S #812", 64.32, "2025-03-14"),
    ("SHELL OIL #204", 41.80, "2025-01-22"),
    ("NETFLIX.COM", 15.99, "2025-05-01"),
    ("CHASE MORTGAGE PYMT", 1850.00, "2025-02-01"),
    ("UBER TRIP", 12.40, "2025-04-18"),
    ("CVS PHARMACY #98", 24.50, "2025-07-03"),
    ("DELTA AIR LINES", 410.00, "2025-08-15"),
    ("PETSMART #55", 38.20, "2025-03-29"),
    ("CHASE CARD PAYMENT", 300.00, "2025-06-25"),
]


def print_prediction(predictor: SklearnPredictor, description: str, amount: float, date: str) -> None:
    result = predictor.predict(description, amount, date)
    print(
        f"{description!r} ${amount:.2f} {date} -> "
        f"{result.category} / {result.subcategory} "
        f"(confidence={result.confidence:.2f}, "
        f"category={result.category_confidence:.2f}, subcategory={result.subcategory_confidence:.2f})"
    )


def main() -> None:
    predictor = SklearnPredictor()

    if len(sys.argv) == 4:
        description, amount, date = sys.argv[1], float(sys.argv[2]), sys.argv[3]
        print_prediction(predictor, description, amount, date)
        return

    for description, amount, date in SANITY_EXAMPLES:
        print_prediction(predictor, description, amount, date)


if __name__ == "__main__":
    main()
