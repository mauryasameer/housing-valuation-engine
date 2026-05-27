import os

import joblib
import pandas as pd


def test_model_pipeline_inference():
    """Integration test to verify that the serialized end-to-end model pipeline

    runs prediction successfully on a mock housing record.
    """
    model_path = "src/data/model_pipeline.joblib"
    if not os.path.exists(model_path):
        # Skip if model artifact is not generated in CI (e.g. if we check before running train)
        return

    pipeline = joblib.load(model_path)

    # Reconstruct a mock raw house record
    raw_input = pd.DataFrame(
        [
            {
                "MSSubClass": 20,
                "LotArea": 9500,
                "OverallQual": 6,
                "OverallCond": 5,
                "YearBuilt": 1970,
                "YearRemodAdd": 1970,
                "MasVnrArea": 0.0,
                "BsmtFinSF1": 400.0,
                "BsmtUnfSF": 500.0,
                "TotalBsmtSF": 900.0,
                "GrLivArea": 1500.0,
                "GarageCars": 2.0,
                "OpenPorchSF": 40.0,
                "GarageYrBlt": 1970,
                "YrSold": 2008,
                "LotShape": "Reg",
                "ExterQual": "TA",
                "BsmtQual": "TA",
                "BsmtExposure": "No",
                "BsmtFinType1": "Unf",
                "HeatingQC": "TA",
                "KitchenQual": "TA",
                "FireplaceQu": "None",
                "GarageFinish": "Unf",
                "BldgType": "1Fam",
                "HouseStyle": "1Story",
                "Fence": "None",
                "LotConfig": "Inside",
                "MasVnrType": "None",
                "SaleCondition": "Normal",
                "MSZoning": "RL",
                "Neighborhood": "NAmes",
                "RoofStyle": "Gable",
                "Exterior1st": "VinylSd",
                "Exterior2nd": "VinylSd",
                "Foundation": "PConc",
                "GarageType": "Attchd",
            }
        ]
    )

    pred = pipeline.predict(raw_input)
    assert len(pred) == 1
    assert pred[0] > 0.0
