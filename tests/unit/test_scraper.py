from unittest.mock import MagicMock, patch

import pandas as pd

EXPECTED_COLUMNS = {
    "region", "bhk", "area_sqft", "floor", "total_floors",
    "age_years", "furnishing", "locality", "parking", "lift",
    "metro_dist_km", "price_inr", "rent_inr",
}

MOCK_HTML = """
<html><body>
<div class="_3bgq4">
  <div class="_2ipzj">3 BHK Flat</div>
  <div class="_1hziv">&#x20B9; 1.2 Cr</div>
  <div class="_2r7aN">1450 sqft</div>
  <div class="_3Cp4n">DLF Phase 1, Gurgaon</div>
  <div class="d-flex">Floor 8 of 15</div>
</div>
</body></html>
"""


def test_parse_listings_returns_dataframe():
    from scripts.scrape_delhi_ncr import parse_listings_html
    result = parse_listings_html(MOCK_HTML, region="Gurgaon")
    assert isinstance(result, pd.DataFrame)


def test_parse_listings_has_required_columns():
    from scripts.scrape_delhi_ncr import parse_listings_html
    result = parse_listings_html(MOCK_HTML, region="Gurgaon")
    for col in EXPECTED_COLUMNS:
        assert col in result.columns, f"Missing column: {col}"


def test_deduplicate_removes_duplicate_rows():
    from scripts.scrape_delhi_ncr import deduplicate
    df = pd.DataFrame([
        {"locality": "DLF Phase 1", "area_sqft": 1100, "floor": 5, "price_inr": 9200000},
        {"locality": "DLF Phase 1", "area_sqft": 1100, "floor": 5, "price_inr": 9200000},
        {"locality": "Sector 62",   "area_sqft": 950,  "floor": 3, "price_inr": 6800000},
    ])
    result = deduplicate(df)
    assert len(result) == 2


def test_fetch_does_not_make_real_http_calls():
    with patch("httpx.Client") as mock_client:
        mock_response = MagicMock()
        mock_response.text = MOCK_HTML
        mock_response.raise_for_status = MagicMock()
        mock_client.return_value.__enter__.return_value.get.return_value = mock_response

        from scripts.scrape_delhi_ncr import fetch_region_listings
        result = fetch_region_listings("Gurgaon", max_pages=1)
        assert isinstance(result, pd.DataFrame)
        mock_client.return_value.__enter__.return_value.get.assert_called()
