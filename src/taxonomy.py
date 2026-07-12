"""Category/subcategory taxonomy, transcribed verbatim from starting-prompt.md."""

CATEGORIES = {
    "Housing & Real Estate": [
        "Mortgage",
        "Rent",
        "Property Taxes",
        "Homeowner's Insurance",
        "HOA fees",
        "Maintenance & repairs",
    ],
    "Healthcare": [
        "Medicare premiums (Parts B, C, D)",
        "Private supplemental health insurance",
        "Dental/vision care",
        "Prescriptions",
        "Out-of-pocket medical costs",
    ],
    "Utilities & Communication": [
        "Electricity",
        "Gas",
        "Water/sewer",
        "Trash",
        "Internet",
        "Cell phone",
        "Cable/streaming services",
    ],
    "Food & Dining": [
        "Groceries",
        "Dining out",
        "Coffee shops",
        "Fast food",
        "Meal delivery services",
        "Snacks & Desserts",
    ],
    "Transportation & Vehicle": [
        "Registration/licensing",
        "Gas/fuel",
        "Vehicle maintenance & repairs",
        "Car payments",
        "Auto insurance",
        "Public transit/ride-sharing",
    ],
    "Entertainment & Leisure": [
        "Hobbies",
        "Event tickets",
        "Gym/club memberships",
        "Classes",
        "Books/Periodicals",
        "Recreational equipment",
        "Apps/services",
    ],
    "Travel & Vacations": [
        "Flights",
        "Lodging",
        "Cruises",
        "Tour packages",
        "Travel insurance",
    ],
    "Personal Care & Apparel": [
        "Clothing",
        "Shoes",
        "Salon/barber visits",
        "Cosmetics/personal hygiene products",
        "Supplements/vitamins",
    ],
    "Household": [
        "Cleaning supplies",
        "Furniture",
        "Appliances",
        "Office Expenses",
        "Misc",
    ],
    "Family & Dependent Support": [
        "Elder care support",
        "Child education/tuition",
        "Childcare",
        "Routine family financial assistance.",
    ],
    "Gifts & Charity": [
        "Charitable donations",
        "Holiday/birthday gifts",
        "Tithing",
    ],
    "Pets": [
        "Vet care",
        "Pet food & supplies",
        "Grooming",
        "Pet insurance",
        "Bird supplies",
    ],
    "Insurance (Non-housing/Non-auto)": [
        "Term or whole life insurance",
        "Disability insurance",
        "Umbrella liability insurance",
        "Long-term care insurance policies",
    ],
    "Debts & Loans": [
        "Student loans",
        "Credit card balances",
        "Personal installment loans",
    ],
}

UNKNOWN_LABEL = "Unknown"


def slugify_category(category: str) -> str:
    """Filesystem/identifier-safe slug for a category name, e.g. for artifact filenames."""
    slug = category.lower()
    for ch in ("&", "(", ")", ",", "/", "'", "."):
        slug = slug.replace(ch, "")
    slug = slug.replace("--", "-")
    return "_".join(slug.split())


if __name__ == "__main__":
    n_categories = len(CATEGORIES)
    n_subcategories = sum(len(v) for v in CATEGORIES.values())
    print(n_categories, n_subcategories)
