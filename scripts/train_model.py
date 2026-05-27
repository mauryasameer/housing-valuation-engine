import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.preprocessing import AligningDummifier, FeatureSelector, HouseFeatureDeriver

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# List of 55 features selected by RFE in the original research
SELECTED_FEATURES = [
    'MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF',
    'GrLivArea', 'GarageCars', 'OpenPorchSF', 'NeworOldGarage', 'Remodelled', 'BuiltRemodelAge',
    'd_BsmtQual', 'd_HeatingQC', 'd_KitchenQual', 'd_FireplaceQu', 'd_HouseStyle', 'd_SaleCondition',
    'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'Neighborhood_BrDale', 'Neighborhood_BrkSide',
    'Neighborhood_CollgCr', 'Neighborhood_Edwards', 'Neighborhood_Gilbert', 'Neighborhood_IDOTRR',
    'Neighborhood_MeadowV', 'Neighborhood_Mitchel', 'Neighborhood_NAmes', 'Neighborhood_NWAmes',
    'Neighborhood_OldTown', 'Neighborhood_SWISU', 'Neighborhood_Sawyer', 'Neighborhood_SawyerW',
    'Neighborhood_Somerst', 'Neighborhood_Timber', 'Exterior1st_CBlock', 'Exterior1st_CemntBd',
    'Exterior1st_Plywood', 'Exterior1st_VinylSd', 'Exterior1st_Wd Sdng', 'Exterior2nd_CBlock',
    'Exterior2nd_CmentBd', 'Exterior2nd_Other', 'Exterior2nd_VinylSd', 'Exterior2nd_Wd Sdng',
    'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'GarageType_Attchd', 'GarageType_BuiltIn',
    'GarageType_Detchd'
]

def main():
    data_path = "src/data/train.csv"
    if not os.path.exists(data_path):
        logger.error("Dataset not found at %s. Ensure you are in the project root.", data_path)
        return

    logger.info("Loading training dataset...")
    df = pd.read_csv(data_path, encoding='latin')

    # Drop outliers as in research notebook (values beyond 98% quantile for LotArea & MasVnrArea)
    lot_area_q98 = df['LotArea'].quantile(0.98)
    mas_vnr_q98 = df['MasVnrArea'].quantile(0.98)
    df_clean = df[(df['LotArea'] <= lot_area_q98) & (df['MasVnrArea'].fillna(0) <= mas_vnr_q98)].copy()

    # Pre-calculate BuiltRemodelAge to filter out rows where age < 0 (as in the notebook)
    # BuiltRemodelAge = YrSold - YearBuilt (if not remodelled) or YrSold - YearRemodAdd
    # Filter: BuiltRemodelAge >= 0
    age = np.where(
        df_clean['YearBuilt'] == df_clean['YearRemodAdd'],
        df_clean['YrSold'] - df_clean['YearBuilt'],
        df_clean['YrSold'] - df_clean['YearRemodAdd']
    )
    df_clean = df_clean[age >= 0].copy()

    # Split into features X and target y
    X = df_clean.drop(columns=['SalePrice'])
    y = df_clean['SalePrice']

    # Step 1: Pre-process X data to create features (fit/transform pipeline)
    # We define preprocessor parts first
    deriver = HouseFeatureDeriver()
    dummifier = AligningDummifier(columns=['MSZoning', 'Neighborhood', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'Foundation', 'GarageType'])
    selector = FeatureSelector(features=SELECTED_FEATURES)
    scaler = StandardScaler()

    # Create the complete feature preprocessing pipeline
    prep_pipeline = Pipeline([
        ('deriver', deriver),
        ('dummifier', dummifier),
        ('selector', selector),
        ('scaler', scaler)
    ])

    logger.info("Splitting dataset into train and test splits...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)

    logger.info("Fitting preprocessing pipeline and transforming training features...")
    X_train_prep = prep_pipeline.fit_transform(X_train)
    X_test_prep = prep_pipeline.transform(X_test)

    # Step 2: Fit regressor with Target Log Transformation
    # We use LassoCV to find the optimal alpha (hyperparameter tuning)
    logger.info("Tuning Lasso regression using cross-validation...")
    # List of alphas from notebook grid search
    alphas = [0.0001, 0.001, 0.0002, 0.0003, 0.0004, 0.0005, 0.01, 0.1, 1.0, 10.0]
    lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42)

    # Wrap regressor in TransformedTargetRegressor to automatically handle target log transformations
    regressor = TransformedTargetRegressor(
        regressor=lasso_cv,
        func=np.log1p,
        inverse_func=np.expm1
    )

    logger.info("Fitting target-transformed Lasso regression model...")
    regressor.fit(X_train_prep, y_train)

    best_alpha = regressor.regressor_.alpha_
    logger.info("Optimal alpha selected by CV: %s", best_alpha)

    # Evaluate the model
    y_pred_train = regressor.predict(X_train_prep)
    y_pred_test = regressor.predict(X_test_prep)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    logger.info("--- Model Evaluation ---")
    logger.info("Train R2: %.4f | Test R2: %.4f", train_r2, test_r2)
    logger.info("Train MSE: %.2e | Test MSE: %.2e", train_mse, test_mse)

    # Initialize SHAP Linear Explainer on training data
    # SHAP LinearExplainer takes the model, and the background data (X_train_prep)
    logger.info("Initializing SHAP LinearExplainer...")
    # Wrap in a standard linear model format for SHAP
    # Since TransformedTargetRegressor wraps the regressor, we extract the underlying fitted Lasso model
    fitted_lasso = regressor.regressor_

    # We pass the fitted model and background training data to calculate attributions on the log scale
    explainer = shap.LinearExplainer(fitted_lasso, X_train_prep)

    # Assemble the final unified pipeline including preprocessing, scaling, and the target-transformed regressor
    # This represents a single end-to-end inference object!
    final_pipeline = Pipeline([
        ('prep', prep_pipeline),
        ('model', regressor)
    ])

    # Save artifacts
    logger.info("Saving trained pipeline and SHAP explainer artifacts...")
    joblib.dump(final_pipeline, "src/data/model_pipeline.joblib")
    joblib.dump(explainer, "src/data/shap_explainer.joblib")

    # Also save training metadata for evaluation in app
    metadata = {
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "train_mse": float(train_mse),
        "test_mse": float(test_mse),
        "X_train_prep": X_train_prep,
        "X_test_prep": X_test_prep,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "features": SELECTED_FEATURES
    }
    joblib.dump(metadata, "src/data/model_metadata.joblib")

    logger.info("Training pipeline and serialization complete!")

if __name__ == "__main__":
    main()
