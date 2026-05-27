import argparse
import logging
import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.utils.delhi_preprocessing import DelhiColumnFinalizer, DelhiFeatureTransformer, DelhiNumericScaler
from src.utils.preprocessing import AligningDummifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
FEATURE_COLS = [
    "bhk", "area_sqft", "floor", "total_floors",
    "age_years", "furnishing", "locality",
    "parking", "lift", "metro_dist_km",
]

# Approximate coordinates of major metro stations across Delhi NCR
METRO_STATIONS = [
    (28.6328, 77.2197),  # Rajiv Chowk
    (28.5529, 77.0587),  # Dwarka Sector 21
    (28.6139, 77.2090),  # New Delhi
    (28.5921, 77.2220),  # Central Secretariat
    (28.6448, 77.2167),  # Kashmere Gate
    (28.5755, 77.3588),  # Noida City Centre
    (28.5630, 77.3356),  # Botanical Garden
    (28.6267, 77.3694),  # Sector 62 Noida
    (28.5355, 77.3910),  # Noida Electronic City
    (28.4594, 77.0726),  # HUDA City Centre Gurgaon
    (28.4419, 77.0988),  # Sector 54 Chowk
    (28.4302, 77.1009),  # Golf Course
    (28.4698, 77.0262),  # Iffco Chowk
    (28.3932, 77.3146),  # NHPC Chowk Faridabad
    (28.4068, 77.3132),  # Faridabad
    (28.6692, 77.4538),  # Ghaziabad
]

REGION_KEYWORDS = {
    "Gurgaon": ["gurgaon", "gurugram"],
    "Noida": ["noida", "greater noida"],
    "Delhi": ["delhi", "new delhi", "dwarka", "rohini", "saket", "vasant kunj",
              "lajpat nagar", "janakpuri"],
    "Faridabad": ["faridabad"],
    "Ghaziabad": ["ghaziabad"],
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return r * 2 * math.asin(math.sqrt(a))


def _nearest_metro_km(lat: float, lon: float) -> float:
    return min(_haversine_km(lat, lon, slat, slon) for slat, slon in METRO_STATIONS)


def _parse_region(address: str) -> str | None:
    addr_lower = address.lower()
    for region, keywords in REGION_KEYWORDS.items():
        if any(kw in addr_lower for kw in keywords):
            return region
    return None


def _parse_locality(address: str) -> str:
    parts = [p.strip() for p in address.split(",")]
    return parts[0] if parts else "Unknown"


def prepare_dataset(raw_path: str) -> pd.DataFrame:
    """Convert Kaggle Delhi_v2.csv format to the expected training schema."""
    df = pd.read_csv(raw_path, encoding="utf-8")
    logger.info("Raw dataset: %d rows, columns: %s", len(df), list(df.columns))

    # Filter to flats only
    df = df[df["type_of_building"] == "Flat"].copy()
    logger.info("After flat filter: %d rows", len(df))

    # Parse region and locality from Address
    df["region"] = df["Address"].apply(_parse_region)
    df["locality"] = df["Address"].apply(_parse_locality)
    df = df[df["region"].notna()].copy()
    logger.info("After region parse: %d rows", len(df))

    # Column mappings
    df["price_inr"] = df["price"]
    df["bhk"] = df["Bedrooms"].fillna(2).astype(int)
    df["area_sqft"] = df["area"]
    df["furnishing"] = df["Furnished_status"].fillna("Semi-Furnished")
    df["age_years"] = df["neworold"].map({"Resale": 10.0, "New Property": 0.0}).fillna(5.0)
    df["parking"] = (df["parking"].fillna(0) > 0).astype(int)
    df["lift"] = (df["Lift"].fillna(0) > 0).astype(int)

    # No floor data — fill with sensible defaults
    df["floor"] = 3
    df["total_floors"] = 10

    # Metro distance from lat/lng
    df["metro_dist_km"] = df.apply(
        lambda r: _nearest_metro_km(r["latitude"], r["longitude"])
        if pd.notna(r["latitude"]) else 2.0,
        axis=1,
    )

    # No rent data in this dataset
    df["rent_inr"] = np.nan

    keep = ["region", "locality", "bhk", "area_sqft", "floor", "total_floors",
            "age_years", "furnishing", "parking", "lift", "metro_dist_km",
            "price_inr", "rent_inr"]
    df = df[keep].dropna(subset=["price_inr", "area_sqft"])

    # Cap localities at top 15 per region to avoid sparse dummy explosion
    top_n = 15
    for region in df["region"].unique():
        mask = df["region"] == region
        top = df.loc[mask, "locality"].value_counts().head(top_n).index
        df.loc[mask & ~df["locality"].isin(top), "locality"] = "Other"

    # Winsorize price at 99th percentile per region — extreme outliers cause expm1 overflow
    for region in df["region"].unique():
        mask = df["region"] == region
        cap = df.loc[mask, "price_inr"].quantile(0.99)
        df.loc[mask, "price_inr"] = df.loc[mask, "price_inr"].clip(upper=cap)

    logger.info("Final prepared dataset: %d rows", len(df))
    return df


def build_prep_pipeline() -> Pipeline:
    return Pipeline([
        ("transformer", DelhiFeatureTransformer()),
        ("dummifier", AligningDummifier(columns=["furnishing", "locality"])),
        ("finalizer", DelhiColumnFinalizer()),
        ("scaler", DelhiNumericScaler()),
    ])


def train_region_mode(df: pd.DataFrame, region: str, mode: str, target_col: str) -> None:
    if target_col not in df.columns or df[target_col].isna().all():
        logger.warning("Skipping %s %s — no %s data", region, mode, target_col)
        return

    df_region = df[(df["region"] == region) & df[target_col].notna()].copy()

    if len(df_region) < 30:
        logger.warning("Skipping %s %s — only %d rows", region, mode, len(df_region))
        return

    logger.info("Training %s | %s (%d rows)", region, mode, len(df_region))

    X = df_region[FEATURE_COLS]
    y = df_region[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

    prep_pipeline = build_prep_pipeline()
    X_train_prep = prep_pipeline.fit_transform(X_train)
    X_test_prep = prep_pipeline.transform(X_test)

    regressor = TransformedTargetRegressor(
        regressor=LassoCV(alphas=ALPHAS, cv=5, random_state=42),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    regressor.fit(X_train_prep, y_train)

    # R² in log space — the correct metric for log-transformed price models
    # (original-space R² is misleading for high-variance price data)
    fitted_lasso = regressor.regressor_
    y_train_log = np.log1p(y_train.values)
    y_test_log = np.log1p(y_test.values)
    train_r2 = r2_score(y_train_log, fitted_lasso.predict(X_train_prep))
    test_r2 = r2_score(y_test_log, fitted_lasso.predict(X_test_prep))

    y_pred_train = regressor.predict(X_train_prep)
    y_pred_test = regressor.predict(X_test_prep)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    logger.info("  Train R2 (log): %.4f | Test R2 (log): %.4f", train_r2, test_r2)

    explainer = shap.LinearExplainer(fitted_lasso, X_train_prep)

    full_pipeline = Pipeline([("prep", prep_pipeline), ("model", regressor)])
    feature_names = prep_pipeline.named_steps["finalizer"].columns_

    os.makedirs("models/delhi_ncr", exist_ok=True)
    tag = f"{region}_{mode}"
    joblib.dump(full_pipeline, f"models/delhi_ncr/model_{tag}.joblib")
    joblib.dump(explainer, f"models/delhi_ncr/shap_{tag}.joblib")
    joblib.dump(
        {
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "train_mse": float(train_mse),
            "test_mse": float(test_mse),
            "features": feature_names,
            "X_train_prep": X_train_prep,
            "X_test_prep": X_test_prep,
            "y_train": y_train.values,
            "y_test": y_test.values,
        },
        f"models/delhi_ncr/metadata_{tag}.joblib",
    )
    logger.info("  Saved artifacts for %s", tag)


def prepare_rent_dataset(rent_path: str) -> pd.DataFrame:
    """Convert andynath/new-delhi-rental-listings CSV to training schema."""
    df = pd.read_csv(rent_path, encoding="utf-8")
    logger.info("Rent dataset: %d rows", len(df))

    # Keep Apartments and Independent Floors
    df = df[df["propertyType"].isin(["Apartment", "Independent Floor"])].copy()

    df["region"] = "Delhi"
    df["locality"] = df["localityName"].fillna("Unknown")
    df["bhk"] = df["bedrooms"].fillna(2).astype(int)
    df["area_sqft"] = df["size_sq_ft"]
    df["furnishing"] = "Semi-Furnished"
    df["age_years"] = 5.0
    df["parking"] = 0
    df["lift"] = 0
    df["floor"] = 3
    df["total_floors"] = 10
    df["metro_dist_km"] = df["closest_mtero_station_km"].fillna(2.0)
    df["rent_inr"] = df["price"]
    df["price_inr"] = np.nan

    keep = ["region", "locality", "bhk", "area_sqft", "floor", "total_floors",
            "age_years", "furnishing", "parking", "lift", "metro_dist_km",
            "price_inr", "rent_inr"]
    df = df[keep].dropna(subset=["rent_inr", "area_sqft"])

    # Cap localities at top 15 and winsorize rent at 99th percentile
    top = df["locality"].value_counts().head(15).index
    df.loc[~df["locality"].isin(top), "locality"] = "Other"
    cap = df["rent_inr"].quantile(0.99)
    df["rent_inr"] = df["rent_inr"].clip(upper=cap)

    logger.info("Final rent dataset: %d rows", len(df))
    return df


def main(only_updated: bool = False) -> None:
    # Accept either base.csv (pre-processed) or the raw Kaggle file
    base_path = "src/data/delhi_ncr/base.csv"
    raw_path = "src/data/delhi_ncr/Delhi_v2.csv"
    rent_path = "src/data/delhi_ncr/rent_delhi.csv"

    if os.path.exists(base_path):
        df_sale = pd.read_csv(base_path, encoding="utf-8")
        logger.info("Loaded %d rows from %s", len(df_sale), base_path)
    elif os.path.exists(raw_path):
        logger.info("base.csv not found — preparing from %s", raw_path)
        df_sale = prepare_dataset(raw_path)
    else:
        logger.error("No sale dataset found. Expected %s or %s", base_path, raw_path)
        df_sale = None

    df_rent = prepare_rent_dataset(rent_path) if os.path.exists(rent_path) else None
    if df_rent is None:
        logger.warning("No rent dataset found at %s — skipping rent models", rent_path)

    with open("configs/delhi_ncr_regions.yaml") as f:
        config = yaml.safe_load(f)

    for region_cfg in config["regions"]:
        if not region_cfg["model_ready"]:
            continue
        region = region_cfg["name"]

        if only_updated:
            live_path = f"src/data/delhi_ncr/{region.lower()}_live.csv"
            if not os.path.exists(live_path):
                continue
            live_df = pd.read_csv(live_path)
            if len(live_df) < 50:
                logger.info("Skipping %s — fewer than 50 new rows", region)
                continue
            if df_sale is not None:
                df_sale = pd.concat([df_sale, pd.read_csv(live_path)], ignore_index=True)

        if df_sale is not None:
            train_region_mode(df_sale, region, "sale", "price_inr")
        if df_rent is not None:
            train_region_mode(df_rent, region, "rent", "rent_inr")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-updated", action="store_true")
    args = parser.parse_args()
    main(only_updated=args.only_updated)
