"""Synthetic transaction dataset generator, driven directly by taxonomy.CATEGORIES.

Also runnable as a script: `uv run python -m src.synthetic_data`.
"""

import random
from datetime import date, timedelta

import pandas as pd

from src import config, taxonomy

# Per-subcategory (merchant description templates, min amount, max amount).
# "{n}" in a template is replaced with a random 3-4 digit store/reference number.
SUBCATEGORY_SPEC = {
    # Housing & Real Estate
    "Mortgage": (["WELLS FARGO HOME MTG", "CHASE MORTGAGE PYMT", "ROCKET MORTGAGE LLC"], 1200, 2800),
    "Rent": (["ZILLOW RENTAL PYMT", "APARTMENTS.COM RENT", "GREYSTAR PROPERTY MGMT"], 1100, 2600),
    "Property Taxes": (["COUNTY TAX COLLECTOR", "PROPERTY TAX ESCROW", "TREASURER TAX OFC"], 400, 3500),
    "Homeowner's Insurance": (["STATE FARM INSURANCE", "ALLSTATE HOME INS", "USAA HOMEOWNERS"], 80, 350),
    "HOA fees": (["HOA MGMT ASSOC #{n}", "ASSOCIA HOA DUES", "COMMUNITY HOA FEE"], 150, 600),
    "Maintenance & repairs": (["ACE HARDWARE #{n}", "HANDYMAN PRO SVC", "ROTO-ROOTER PLUMBING"], 60, 900),
    # Healthcare
    "Medicare premiums (Parts B, C, D)": (["MEDICARE PREMIUM", "CMS MEDICARE PART B", "SSA MEDICARE DEDUCT"], 60, 220),
    "Private supplemental health insurance": (["AETNA SUPPLEMENTAL", "UNITEDHEALTH MEDIGAP", "CIGNA SUPP HEALTH"], 100, 400),
    "Dental/vision care": (["ASPEN DENTAL #{n}", "LENSCRAFTERS #{n}", "BRIGHT SMILES DENTAL"], 40, 500),
    "Prescriptions": (["CVS PHARMACY #{n}", "WALGREENS RX #{n}", "RITE AID PHARMACY"], 10, 180),
    "Out-of-pocket medical costs": (["URGENT CARE CLINIC", "QUEST DIAGNOSTICS", "REGIONAL MEDICAL CTR"], 20, 600),
    # Utilities & Communication
    "Electricity": (["PG&E ELECTRIC BILL", "DUKE ENERGY PAYMT", "CON EDISON UTILITY"], 60, 280),
    "Gas": (["NATIONAL GRID GAS", "SOCALGAS PAYMENT", "CENTERPOINT ENERGY"], 25, 180),
    "Water/sewer": (["CITY WATER UTILITY", "MUNICIPAL WATER DEPT", "AQUA WATER SEWER"], 30, 140),
    "Trash": (["WASTE MANAGEMENT INC", "REPUBLIC SERVICES", "CITY TRASH COLLECTION"], 20, 70),
    "Internet": (["COMCAST XFINITY", "SPECTRUM INTERNET", "AT&T FIBER INTERNET"], 45, 110),
    "Cell phone": (["VERIZON WIRELESS", "T-MOBILE PAYMENT", "AT&T MOBILITY"], 35, 160),
    "Cable/streaming services": (["NETFLIX.COM", "HULU SUBSCRIPTION", "SPECTRUM CABLE TV"], 8, 90),
    # Food & Dining
    "Groceries": (["TRADER JOE'S #{n}", "WHOLE FOODS MKT #{n}", "SAFEWAY #{n}", "KROGER #{n}"], 15, 220),
    "Dining out": (["OLIVE GARDEN #{n}", "THE CHEESECAKE FACTORY", "LOCAL BISTRO #{n}"], 20, 140),
    "Coffee shops": (["STARBUCKS #{n}", "PEET'S COFFEE #{n}", "DUNKIN' #{n}"], 3, 12),
    "Fast food": (["MCDONALD'S #{n}", "CHIPOTLE #{n}", "TACO BELL #{n}", "CHICK-FIL-A #{n}"], 5, 25),
    "Meal delivery services": (["DOORDASH*ORDER", "UBER EATS", "GRUBHUB ORDER #{n}"], 15, 60),
    "Snacks & Desserts": (["7-ELEVEN #{n}", "BASKIN ROBBINS #{n}", "COLD STONE CREAMERY"], 2, 20),
    # Transportation & Vehicle
    "Registration/licensing": (["DMV VEHICLE REG", "STATE DMV FEE", "COUNTY VEHICLE LICENSE"], 40, 300),
    "Gas/fuel": (["SHELL OIL #{n}", "CHEVRON #{n}", "EXXONMOBIL #{n}"], 25, 90),
    "Vehicle maintenance & repairs": (["JIFFY LUBE #{n}", "MIDAS AUTO REPAIR", "FIRESTONE COMPLETE AUTO"], 40, 900),
    "Car payments": (["TOYOTA FINANCIAL SVC", "ALLY AUTO PAYMENT", "HONDA FINANCIAL"], 250, 650),
    "Auto insurance": (["GEICO AUTO INSURANCE", "PROGRESSIVE INSURANCE", "STATE FARM AUTO"], 90, 260),
    "Public transit/ride-sharing": (["UBER TRIP", "LYFT RIDE", "METRO TRANSIT FARE"], 3, 45),
    # Entertainment & Leisure
    "Hobbies": (["MICHAELS STORES #{n}", "HOBBY LOBBY #{n}", "GUITAR CENTER #{n}"], 10, 200),
    "Event tickets": (["TICKETMASTER EVENT", "STUBHUB TICKETS", "AXS TICKET PURCHASE"], 25, 300),
    "Gym/club memberships": (["PLANET FITNESS #{n}", "LA FITNESS MEMBERSHIP", "EQUINOX CLUB DUES"], 20, 180),
    "Classes": (["MASTERCLASS.COM", "LOCAL COMMUNITY COLLEGE", "SKILLSHARE SUBSCRIPTION"], 15, 400),
    "Books/Periodicals": (["BARNES & NOBLE #{n}", "AMAZON.COM BOOKS", "NEW YORK TIMES SUB"], 8, 60),
    "Recreational equipment": (["REI CO-OP #{n}", "DICK'S SPORTING GOODS", "ACADEMY SPORTS #{n}"], 20, 500),
    "Apps/services": (["APPLE.COM/BILL", "GOOGLE PLAY APPS", "SPOTIFY PREMIUM"], 3, 25),
    # Travel & Vacations
    "Flights": (["DELTA AIR LINES", "UNITED AIRLINES", "SOUTHWEST AIRLINES"], 120, 900),
    "Lodging": (["MARRIOTT HOTELS #{n}", "AIRBNB PAYMENT", "HILTON HOTELS #{n}"], 90, 600),
    "Cruises": (["ROYAL CARIBBEAN CRUISE", "CARNIVAL CRUISE LINE", "NORWEGIAN CRUISE LINE"], 400, 3000),
    "Tour packages": (["EXPEDIA TOUR PACKAGE", "VIATOR TOURS", "TRIPADVISOR EXPERIENCES"], 60, 800),
    "Travel insurance": (["ALLIANZ TRAVEL INS", "WORLD NOMADS INSURANCE", "TRAVELEX INSURANCE"], 25, 150),
    # Personal Care & Apparel
    "Clothing": (["GAP #{n}", "OLD NAVY #{n}", "NORDSTROM #{n}"], 15, 220),
    "Shoes": (["DSW #{n}", "FOOT LOCKER #{n}", "NIKE STORE #{n}"], 25, 180),
    "Salon/barber visits": (["GREAT CLIPS #{n}", "LOCAL BARBER SHOP", "SUPERCUTS #{n}"], 15, 120),
    "Cosmetics/personal hygiene products": (["SEPHORA #{n}", "ULTA BEAUTY #{n}", "TARGET #{n}"], 8, 90),
    "Supplements/vitamins": (["GNC #{n}", "VITAMIN SHOPPE #{n}", "IHERB.COM"], 10, 70),
    # Household
    "Cleaning supplies": (["TARGET #{n}", "COSTCO WHOLESALE #{n}", "THE HOME DEPOT #{n}"], 8, 60),
    "Furniture": (["IKEA #{n}", "ASHLEY FURNITURE", "WAYFAIR.COM"], 60, 2000),
    "Appliances": (["BEST BUY #{n}", "LOWE'S #{n}", "HOME DEPOT APPLIANCES"], 100, 1800),
    "Office Expenses": (["STAPLES #{n}", "OFFICE DEPOT #{n}", "AMAZON.COM OFFICE"], 10, 150),
    "Misc": (["TARGET #{n}", "WALMART #{n}", "AMAZON.COM"], 5, 100),
    # Family & Dependent Support
    "Elder care support": (["HOME INSTEAD SENIOR CARE", "COMFORT KEEPERS", "ASSISTED LIVING FACILITY"], 100, 2500),
    "Child education/tuition": (["PRIVATE SCHOOL TUITION", "MONTESSORI ACADEMY", "UNIVERSITY TUITION PYMT"], 200, 5000),
    "Childcare": (["BRIGHT HORIZONS DAYCARE", "KINDERCARE LEARNING CTR", "LOCAL DAYCARE CENTER"], 150, 1200),
    "Routine family financial assistance.": (["ZELLE TRANSFER TO FAMILY", "VENMO FAMILY SUPPORT", "WESTERN UNION TRANSFER"], 50, 800),
    # Gifts & Charity
    "Charitable donations": (["RED CROSS DONATION", "UNITED WAY DONATION", "ST JUDE DONATION"], 10, 500),
    "Holiday/birthday gifts": (["AMAZON.COM GIFT", "TARGET #{n}", "ETSY.COM ORDER"], 15, 200),
    "Tithing": (["FIRST BAPTIST CHURCH", "ST MARY'S PARISH TITHE", "COMMUNITY CHURCH GIVING"], 20, 400),
    # Pets
    "Vet care": (["BANFIELD PET HOSPITAL", "VCA ANIMAL HOSPITAL", "LOCAL VETERINARY CLINIC"], 40, 700),
    "Pet food & supplies": (["PETSMART #{n}", "PETCO #{n}", "CHEWY.COM"], 10, 120),
    "Grooming": (["PETSMART GROOMING #{n}", "LOCAL PET GROOMER", "PETCO GROOMING SALON"], 20, 100),
    "Pet insurance": (["TRUPANION PET INSURANCE", "HEALTHY PAWS INSURANCE", "NATIONWIDE PET INS"], 15, 65),
    "Bird supplies": (["PETCO BIRD SUPPLIES", "WILD BIRD CENTER #{n}", "CHEWY.COM BIRD"], 8, 70),
    # Insurance (Non-housing/Non-auto)
    "Term or whole life insurance": (["NORTHWESTERN MUTUAL", "PRUDENTIAL LIFE INS", "MASSMUTUAL LIFE"], 30, 250),
    "Disability insurance": (["METLIFE DISABILITY", "GUARDIAN DISABILITY INS", "UNUM DISABILITY"], 20, 150),
    "Umbrella liability insurance": (["USAA UMBRELLA POLICY", "STATE FARM UMBRELLA", "TRAVELERS UMBRELLA"], 15, 60),
    "Long-term care insurance policies": (["GENWORTH LTC INSURANCE", "JOHN HANCOCK LTC", "MUTUAL OF OMAHA LTC"], 60, 300),
    # Debts & Loans
    "Student loans": (["NAVIENT LOAN PYMT", "NELNET STUDENT LOAN", "SALLIE MAE PAYMENT"], 100, 600),
    "Credit card balances": (["CHASE CARD PAYMENT", "AMEX BILL PAYMENT", "CAPITAL ONE PAYMENT"], 50, 1500),
    "Personal installment loans": (["SOFI PERSONAL LOAN", "LENDINGCLUB PAYMENT", "UPSTART LOAN PYMT"], 80, 500),
}


def _random_date(start: date, end: date, rng: random.Random) -> date:
    delta_days = (end - start).days
    return start + timedelta(days=rng.randint(0, delta_days))


def generate_synthetic_dataset(n_per_subcategory: int = 30, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic transactions DataFrame with columns:
    date, description, amount, category, subcategory.

    Driven directly by taxonomy.CATEGORIES so the generator can't drift out of
    sync with the taxonomy. Deliberately injects a handful of blank-label rows
    so the drop-report path in data_loader.drop_unlabeled is exercised from
    the first run.
    """
    rng = random.Random(seed)
    start_date, end_date = date(2024, 1, 1), date(2025, 12, 31)

    rows = []
    for category, subcategories in taxonomy.CATEGORIES.items():
        for subcategory in subcategories:
            templates, amount_min, amount_max = SUBCATEGORY_SPEC[subcategory]
            for _ in range(n_per_subcategory):
                template = rng.choice(templates)
                description = template.format(n=rng.randint(100, 9999)) if "{n}" in template else template
                amount = round(rng.uniform(amount_min, amount_max), 2)
                txn_date = _random_date(start_date, end_date, rng)
                rows.append(
                    {
                        "date": txn_date.isoformat(),
                        "description": description,
                        "amount": amount,
                        "category": category,
                        "subcategory": subcategory,
                    }
                )

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Inject ~20-30 blank-label rows to exercise the drop-report path.
    n_blank = rng.randint(20, 30)
    blank_indices = rng.sample(range(len(df)), n_blank)
    for idx in blank_indices:
        if rng.random() < 0.5:
            df.loc[idx, "category"] = ""
            df.loc[idx, "subcategory"] = ""
        else:
            df.loc[idx, "subcategory"] = ""

    return df


def main() -> None:
    df = generate_synthetic_dataset()
    config.SYNTHETIC_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.SYNTHETIC_DATA_PATH, index=False)
    print(f"Wrote {len(df)} rows to {config.SYNTHETIC_DATA_PATH}")


if __name__ == "__main__":
    main()
