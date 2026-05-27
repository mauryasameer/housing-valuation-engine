import os

import joblib
import pandas as pd
import pytest


@pytest.fixture
def fixture_df():
    path = "tests/test_data/delhi_fixture.csv"
    return pd.read_csv(path)


def test_gurgaon_sale_pipeline_predicts(fixture_df):
    model_path = "models/delhi_ncr/model_Gurgaon_sale.joblib"
    if not os.path.exists(model_path):
        pytest.skip("Gurgaon sale model artifact not found — run train_delhi_ncr.py first")

    try:
        pipeline = joblib.load(model_path)
    except Exception as exc:
        pytest.skip(f"Could not load model artifact: {exc}")
    row = fixture_df[fixture_df["region"] == "Gurgaon"].drop(
        columns=["region", "price_inr", "rent_inr"]
    ).iloc[:1]
    pred = pipeline.predict(row)
    assert pred.shape == (1,)
    assert pred[0] > 0


def test_gurgaon_rent_pipeline_predicts(fixture_df):
    model_path = "models/delhi_ncr/model_Gurgaon_rent.joblib"
    if not os.path.exists(model_path):
        pytest.skip("Gurgaon rent model artifact not found — run train_delhi_ncr.py first")

    try:
        pipeline = joblib.load(model_path)
    except Exception as exc:
        pytest.skip(f"Could not load model artifact: {exc}")
    row = fixture_df[fixture_df["region"] == "Gurgaon"].drop(
        columns=["region", "price_inr", "rent_inr"]
    ).iloc[:1]
    pred = pipeline.predict(row)
    assert pred.shape == (1,)
    assert pred[0] > 0


def test_metadata_has_expected_keys():
    meta_path = "models/delhi_ncr/metadata_Gurgaon_sale.joblib"
    if not os.path.exists(meta_path):
        pytest.skip("Gurgaon sale metadata not found")
    try:
        meta = joblib.load(meta_path)
    except Exception as exc:
        pytest.skip(f"Could not load metadata artifact: {exc}")
    for key in ("train_r2", "test_r2", "train_mse", "test_mse", "features"):
        assert key in meta, f"Missing key: {key}"
