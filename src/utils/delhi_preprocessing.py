import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

NUMERIC_FEATURES = [
    "bhk", "area_sqft", "floor", "total_floors",
    "age_years", "metro_dist_km", "parking", "lift",
]
CATEGORICAL_FEATURES = ["furnishing", "locality"]
INT_FEATURES = ["bhk", "floor", "total_floors", "parking", "lift"]


class DelhiFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.medians_: dict = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in NUMERIC_FEATURES:
            if col in X.columns:
                self.medians_[col] = X[col].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in NUMERIC_FEATURES:
            if col in X.columns:
                fill = self.medians_.get(col, 0.0)
                X[col] = X[col].fillna(fill if not pd.isna(fill) else 0.0)
        for col in CATEGORICAL_FEATURES:
            if col in X.columns:
                X[col] = X[col].fillna("Unknown")
        for col in INT_FEATURES:
            if col in X.columns:
                X[col] = X[col].astype(int)
        return X


class DelhiColumnFinalizer(BaseEstimator, TransformerMixin):
    """Stores the fitted column list and ensures consistent column order at inference."""

    def __init__(self):
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns_:
            if col not in X.columns:
                X[col] = 0
        return X[self.columns_]


class DelhiNumericScaler(BaseEstimator, TransformerMixin):
    """StandardScaler applied only to numeric columns; dummy columns left as binary 0/1.

    Scaling dummy columns with StandardScaler creates extreme z-scores for sparse
    categories (e.g., a column that is 1% occupied gets z-score ~10 on the 1s),
    which causes expm1 to overflow during log-scale prediction.
    """

    def __init__(self):
        self.numeric_cols_: list[str] = []
        self.means_: dict[str, float] = {}
        self.stds_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y=None):
        self.numeric_cols_ = [c for c in NUMERIC_FEATURES if c in X.columns]
        for col in self.numeric_cols_:
            self.means_[col] = float(X[col].mean())
            std = float(X[col].std())
            self.stds_[col] = std if std > 0 else 1.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.numeric_cols_:
            X[col] = (X[col] - self.means_[col]) / self.stds_[col]
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features) if input_features is not None else np.array([])
