import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {
            "bhk": 2, "area_sqft": 1100.0, "floor": 5, "total_floors": 15,
            "age_years": 3.0, "furnishing": "Furnished", "locality": "DLF Phase 1",
            "parking": 1, "lift": 1, "metro_dist_km": 0.8,
        },
        {
            "bhk": 3, "area_sqft": None, "floor": 8, "total_floors": 20,
            "age_years": 1.0, "furnishing": "Semi-Furnished", "locality": "Golf Course Road",
            "parking": 1, "lift": 1, "metro_dist_km": 1.2,
        },
    ])


def test_transformer_fills_missing_area(sample_df):
    from src.utils.delhi_preprocessing import DelhiFeatureTransformer
    t = DelhiFeatureTransformer()
    t.fit(sample_df)
    out = t.transform(sample_df)
    assert out["area_sqft"].isna().sum() == 0


def test_transformer_casts_types(sample_df):
    from src.utils.delhi_preprocessing import DelhiFeatureTransformer
    t = DelhiFeatureTransformer()
    t.fit(sample_df)
    out = t.transform(sample_df)
    assert out["bhk"].dtype == np.int64 or out["bhk"].dtype == int
    assert out["parking"].dtype == np.int64 or out["parking"].dtype == int
    assert out["lift"].dtype == np.int64 or out["lift"].dtype == int


def test_column_finalizer_stores_columns(sample_df):
    from src.utils.delhi_preprocessing import DelhiColumnFinalizer
    from src.utils.preprocessing import AligningDummifier

    dummifier = AligningDummifier(columns=["furnishing", "locality"])
    df_dummy = dummifier.fit_transform(sample_df)

    finalizer = DelhiColumnFinalizer()
    finalizer.fit(df_dummy)
    assert len(finalizer.columns_) > 0
    assert "bhk" in finalizer.columns_
    assert "furnishing" not in finalizer.columns_


def test_column_finalizer_pads_missing_column(sample_df):
    from src.utils.delhi_preprocessing import DelhiColumnFinalizer
    from src.utils.preprocessing import AligningDummifier

    dummifier = AligningDummifier(columns=["furnishing", "locality"])
    df_dummy = dummifier.fit_transform(sample_df)

    finalizer = DelhiColumnFinalizer()
    finalizer.fit(df_dummy)

    df_missing = df_dummy.drop(columns=[df_dummy.columns[-1]])
    out = finalizer.transform(df_missing)
    assert list(out.columns) == finalizer.columns_


def test_full_prep_pipeline_returns_array(sample_df):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from src.utils.delhi_preprocessing import DelhiColumnFinalizer, DelhiFeatureTransformer
    from src.utils.preprocessing import AligningDummifier

    pipe = Pipeline([
        ("transformer", DelhiFeatureTransformer()),
        ("dummifier", AligningDummifier(columns=["furnishing", "locality"])),
        ("finalizer", DelhiColumnFinalizer()),
        ("scaler", StandardScaler()),
    ])
    result = pipe.fit_transform(sample_df)
    assert result.shape[0] == 2
    assert result.shape[1] > 8
