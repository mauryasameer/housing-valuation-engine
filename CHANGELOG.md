# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-05-25

### Added
- Delhi NCR flat price predictor tab with Layout C UI: sidebar region list, interactive Folium map (CartoDB dark tiles, green/grey pins by model availability), and Buy/Rent prediction form.
- Per-region LassoCV models (sale price + monthly rent) for Gurgaon, Noida, and Delhi. Faridabad and Ghaziabad unlock as data becomes available.
- `configs/delhi_ncr_regions.yaml` as single source of truth for region availability — drives sidebar, map pins, locality dropdowns, and model loading.
- `src/utils/delhi_preprocessing.py`: `DelhiFeatureTransformer` (median imputation, type casting) and `DelhiColumnFinalizer` (consistent column alignment at inference).
- `src/services/delhi_mrm_service.py`: thin re-export of MRM diagnostics for the Delhi NCR tab.
- `scripts/train_delhi_ncr.py`: trains one LassoCV model per region per mode, saves `.joblib` artifacts to `models/delhi_ncr/`. Supports `--only-updated` for the monthly refresh workflow.
- `scripts/scrape_delhi_ncr.py`: scrapes 99acres listings per region with sha256 deduplication.
- `.github/workflows/delhi_data_refresh.yml`: monthly cron — scrape → retrain updated regions → commit artifacts to `data/refresh-YYYY-MM` → auto-PR to `dev`. Opens a GitHub issue on failure.
- Unit tests: `tests/unit/test_delhi_preprocessing.py` (5 tests), `tests/unit/test_scraper.py` (4 tests, no live HTTP).
- Integration tests: `tests/integration/test_delhi_pipeline.py` (3 tests, skip gracefully when artifacts absent).

[1.1.0]: https://github.com/mauryasameer/Regression_analysis/compare/v1.0.1...v1.1.0

## [1.0.1] - 2026-05-24

### Fixed
- Pinned ruff to `==0.15.14` in `requirements.txt` and CI to prevent local/CI lint divergence.
- Added Python 3.12 to CI test matrix.
- Consolidated `requirements-dev.txt` into `requirements.txt`; removed redundant file.
- Removed committed raw data files (`train.csv`, `bck.png`) and added them to `.gitignore`.

[1.0.1]: https://github.com/mauryasameer/Regression_analysis/compare/v1.0.0...v1.0.1

## [1.0.0] - 2026-05-24

### Added
- Created production-grade machine learning pipelines using scikit-learn `Pipeline` and `ColumnTransformer` to prevent data leakage.
- Implemented **Model Risk Management (MRM)** statistical diagnostic suite (VIF, Breusch-Pagan, Durbin-Watson, Jarque-Bera tests).
- Integrated local LLM capability using Ollama (`llama3.1:8b`) via the **Provider Abstraction Pattern**.
- Established **Cybersecurity** defenses (input sanitization, prompt injection safeguards, XSS output sanitization).
- Developed a root-level Streamlit web dashboard (`app.py`) featuring real-time prediction, interactive SHAP waterfall explanations, and an MRM governance summary.
- Moved research notebook into `notebooks/` directory.
