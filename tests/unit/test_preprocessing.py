import numpy as np
import pandas as pd

from src.utils.preprocessing import HouseFeatureDeriver


def test_house_feature_deriver():
    """Unit test to verify that HouseFeatureDeriver processes numerical values,

    creates derived features, and handles categorical ordinals.
    """
    raw_df = pd.DataFrame(
        [
            {
                "LotFrontage": np.nan,
                "MasVnrArea": np.nan,
                "GarageYrBlt": 1995,
                "YearBuilt": 1995,
                "YearRemodAdd": 2005,
                "YrSold": 2008,
                "LotShape": "Reg",
                "ExterQual": "Ex",
            }
        ]
    )

    deriver = HouseFeatureDeriver()
    deriver.fit(raw_df)
    transformed = deriver.transform(raw_df)

    # Verify column mappings
    assert transformed["LotFrontage"].values[0] == 0.0
    assert transformed["NeworOldGarage"].values[0] == 0
    assert transformed["Remodelled"].values[0] == 1
    assert transformed["BuiltRemodelAge"].values[0] == 3
    assert transformed["d_LotShape"].values[0] == 3
    assert transformed["d_ExterQual"].values[0] == 5
