I want to build a supervised ML system that classifies personal finance transactions
into a fixed set of categories and subcategories (e.g. "Food > Groceries",
"Transport > Rideshare"). This is for learning purposes — I want to understand the
full pipeline, not just get a working black box.

Goals:
1. Train a baseline text classification model on labeled transaction data
   (description/merchant string + amount + day-of-week as features).
2. Serve the trained model locally via a FastAPI inference API.
3. Structure the code so I can later swap in a transformer-based model
   (e.g. DistilBERT or FinBERT) without rewriting the serving layer.

Requirements:
- Python, using scikit-learn for the baseline (TF-IDF or char n-grams + Logistic
  Regression/LinearSVC), with a shared preprocessing module used by both training
  and serving so there's no train/serve skew.
- Two-level classification: predict top-level category first, then subcategory
  (conditioned on category or with a category-specific classifier).
- FastAPI app with a /predict endpoint (accepts a transaction description + amount
  + date, returns category/subcategory + confidence) and /health endpoint.
  Runnable locally with uvicorn.
- Model artifacts saved via joblib, loaded once at API startup.
- A training script that outputs evaluation metrics (per-class precision/recall,
  confusion matrix) so I can judge label quality and model performance.
- Basic project structure: data/, src/ (train.py, features.py, evaluate.py),
  serve/ (main.py), models/, and a requirements.txt.

  Categories and subcategories:
-  Housing & Real Estate
    - Mortgage
    - Rent
    - Property Taxes
    - Homeowner's Insurance
    - HOA fees
    - Maintenance & repairs

- Healthcare
    - Medicare premiums (Parts B, C, D)
    - Private supplemental health insurance
    - Dental/vision care
    - Prescriptions
    - Out-of-pocket medical costs
- Utilities & Communication
    - Electricity
    - Gas
    - Water/sewer
    - Trash
    - Internet
    - Cell phone
    - Cable/streaming services
- Food & Dining
    - Groceries
    - Dining out
    - Coffee shops
    - Fast food
    - Meal delivery services
    - Snacks & Desserts
- Transportation & Vehicle
    - Registration/licensing
    - Gas/fuel
    - Vehicle maintenance & repairs
    - Car payments
    - Auto insurance
    - Public transit/ride-sharing
- Entertainment & Leisure
    - Hobbies
    - Event tickets
    - Gym/club memberships
    - Classes
    - Books/Periodicals
    - Recreational equipment
    - Apps/services
- Travel & Vacations
    - Flights
    - Lodging
    - Cruises
    - Tour packages
    - Travel insurance
-  Personal Care & Apparel
    - Clothing
    - Shoes
    - Salon/barber visits
    - Cosmetics/personal hygiene products
    - Supplements/vitamins
- Household
    - Cleaning supplies
    - Furniture
    - Appliances
    - Office Expenses
    - Misc
- Family & Dependent Support
    - Elder care support
    - Child education/tuition
    - Childcare
    - Routine family financial assistance.
- Gifts & Charity
    - Charitable donations
    - Holiday/birthday gifts
    - Tithing
- Pets
    - Vet care
    - Pet food & supplies
    - Grooming
    - Pet insurance
    - Bird supplies
- Insurance (Non-housing/Non-auto)
    - Term or whole life insurance
    - Disability insurance
    - Umbrella liability insurance
    - Long-term care insurance policies
- Debts & Loans
    - Student loans
    - Credit card balances
    - Personal installment loans

Note: if category or subcategory is missing, note it as Unknown in the label output.


Before processing:
1. Ask me about my category/subcategory taxonomy and what my labeled data looks
   like (columns, format, size) — don't assume a schema.
2. Propose the project structure and confirm it with me.
3. Then scaffold the code incrementally, starting with the training pipeline
   on a small sample/synthetic dataset so I can validate the approach before
   plugging in my real data.

Then, create a comprehensive, multi-step plan that covers:
1. Implementation of the preprocessing pipeline for the data, ensuring it aligns with
   the category/subcategory taxonomy and handles missing/unknown labels.
2. Training the model on a sample/synthetic dataset, fine tuning it with a small sample of real data, and validating predictions against expected outputs.
3. Iterating on preprocessing, feature engineering, and model design as
   needed to improve accuracy.
4. Once validated, scale to the full dataset and integrate into the
   serving pipeline.

I've completed the DeepLearning.ai ML specialization, so I understand the ML
fundamentals — feel free to be technical, but explain design decisions
(especially around preprocessing pipelines and avoiding train/serve skew) as
you go, since I'm newer to the MLOps/serving side.