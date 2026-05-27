import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HouseFeatureDeriver(BaseEstimator, TransformerMixin):
    """Custom transformer to fill missing values, create derived features,

    map ordinal categories, and drop redundant columns.
    """

    def __init__(self):
        self.medians_ = {}

    def fit(self, X, y=None):
        # Calculate medians for numerical columns to use during transform
        for col in ["LotFrontage", "MasVnrArea"]:
            if col in X.columns:
                self.medians_[col] = X[col].median()
        return self

    def transform(self, X):
        X = X.copy()

        # 1. Fill numeric NaNs
        for col in ["LotFrontage", "MasVnrArea"]:
            if col in X.columns:
                fill_val = self.medians_.get(col, 0.0)
                if pd.isna(fill_val):
                    fill_val = 0.0
                X[col] = X[col].fillna(fill_val)

        # 2. Fill categorical NaNs
        nan_cols_cats = [
            "MasVnrType",
            "BsmtQual",
            "BsmtCond",
            "BsmtExposure",
            "BsmtFinType1",
            "BsmtFinType2",
            "FireplaceQu",
            "GarageType",
            "GarageFinish",
            "GarageQual",
            "GarageCond",
            "Fence",
            "Electrical",
        ]
        for col in nan_cols_cats:
            if col in X.columns:
                X[col] = X[col].fillna("None")

        # 3. Create derived features
        # Garage Yr Built
        if "GarageYrBlt" in X.columns:
            garage_yr = X["GarageYrBlt"].fillna(0).astype(int)
            X["NeworOldGarage"] = garage_yr.apply(lambda r: 0 if r == 0 or (1900 <= r < 2000) else 1)
        else:
            X["NeworOldGarage"] = 0

        # Remodelled and Age
        if "YearBuilt" in X.columns and "YearRemodAdd" in X.columns:
            X["Remodelled"] = (X["YearBuilt"] < X["YearRemodAdd"]).astype(int)

            # Age calculation: use YrSold or default to 2008 (median sold year) if missing
            yr_sold = X["YrSold"] if "YrSold" in X.columns else 2008

            # Vectorized age computation
            X["BuiltRemodelAge"] = np.where(
                X["YearBuilt"] == X["YearRemodAdd"],
                yr_sold - X["YearBuilt"],
                yr_sold - X["YearRemodAdd"],
            )
            # Ensure age is non-negative
            X["BuiltRemodelAge"] = np.clip(X["BuiltRemodelAge"], 0, None)
        else:
            X["Remodelled"] = 0
            X["BuiltRemodelAge"] = 0

        # 4. Ordinal mappings
        mappings = {
            "LotShape": {"Reg": 3, "IR1": 2, "IR2": 1, "IR3": 0},
            "ExterQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "BsmtQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0},
            "BsmtFinType1": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0},
            "HeatingQC": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "KitchenQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "FireplaceQu": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "GarageFinish": {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0},
            "BldgType": {"Twnhs": 5, "TwnhsE": 4, "Duplex": 3, "2fmCon": 2, "1Fam": 1, "None": 0},
            "HouseStyle": {
                "SLvl": 8,
                "SFoyer": 7,
                "2.5Fin": 6,
                "2.5Unf": 5,
                "2Story": 4,
                "1.5Fin": 3,
                "1.5Unf": 2,
                "1Story": 1,
                "None": 0,
            },
            "Fence": {"GdPrv": 4, "GdWo": 3, "MnPrv": 2, "MnWw": 1, "None": 0},
            "LotConfig": {"Inside": 5, "Corner": 4, "CulDSac": 3, "FR2": 2, "FR3": 1, "None": 0},
            "MasVnrType": {"BrkCmn": 1, "BrkFace": 1, "CBlock": 1, "Stone": 1, "None": 0},
            "SaleCondition": {
                "Normal": 1,
                "Partial": 1,
                "Abnorml": 0,
                "Family": 0,
                "Alloca": 0,
                "AdjLand": 0,
                "None": 0,
            },
        }

        for col, mapping in mappings.items():
            if col in X.columns:
                X[f"d_{col}"] = X[col].map(mapping).fillna(0).astype(int)

        # Drop raw categorical columns we mapped from
        mapped_cols = list(mappings.keys())
        # Drop redundant columns that cause multicollinearity or are represented by derived features
        redundant_cols = [
            "Id",
            "YearBuilt",
            "YearRemodAdd",
            "YrSold",
            "GarageYrBlt",
            "TotRmsAbvGrd",
            "GarageArea",
            "MoSold",
        ]
        drop_cols = [c for c in mapped_cols + redundant_cols if c in X.columns]
        X = X.drop(columns=drop_cols)

        return X


class AligningDummifier(BaseEstimator, TransformerMixin):
    """Custom one-hot encoder that aligns dummy columns with those seen

    during training, preventing feature mismatch errors in production.
    """

    def __init__(self, columns=None):
        self.columns = columns
        self.dummy_columns_ = []

    def fit(self, X, y=None):
        # Find all dummy columns generated during fit
        X_dummies = pd.get_dummies(X, columns=self.columns, drop_first=True)
        self.dummy_columns_ = [c for c in X_dummies.columns if any(c.startswith(f"{col}_") for col in self.columns)]
        return self

    def transform(self, X):
        X_dummies = pd.get_dummies(X, columns=self.columns, drop_first=True)

        # Ensure all columns from training are present (filling with 0 if missing)
        for col in self.dummy_columns_:
            if col not in X_dummies.columns:
                X_dummies[col] = 0

        # Ensure no extra columns exist (dropping if not seen in training)
        current_dummies = [c for c in X_dummies.columns if any(c.startswith(f"{col}_") for col in self.columns)]
        drop_cols = [c for c in current_dummies if c not in self.dummy_columns_]
        X_dummies = X_dummies.drop(columns=drop_cols)

        return X_dummies


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Selects and orders the exact set of features required by the model."""

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Add any missing features with default 0.0
        for f in self.features:
            if f not in X.columns:
                X[f] = 0.0
        # Return only the requested features in correct order
        return X[self.features]
