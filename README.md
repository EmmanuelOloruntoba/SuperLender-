# SuperLender Credit Risk Project - Final Handoff

This bundle contains the audited Streamlit application and the completed FastAPI production-inference layer.

## Project architecture

```text
app.py / Streamlit
    ├── Data & EDA
    ├── Model Training
    │      └── current training selection in session state
    ├── Risk Assessment
    │      ├── Production model
    │      └── Current training selection
    └── Batch Intelligence
           ├── Production model
           └── Current training selection

FastAPI
    ├── api/schemas.py
    ├── api/main.py
    └── src/
          ├── feature_engineer.py  <-- keep the current working project file
          ├── model_manager.py
          └── risk_prediction_service.py

artifacts/
    ├── superlender_hgb_pipeline.joblib
    └── model_metadata.json
```

## Important integration note

The working project already has a canonical `src/feature_engineer.py`. Keep that exact working file in the project. It is intentionally not replaced by this bundle because the current local project source is not available in this handoff environment.

The FastAPI layer imports it as `from src.feature_engineer import FeatureEngineer` and expects its `MODEL_FEATURES` contract and `build_features(...)` method.

## Production model

Production inference uses the frozen saved pipeline:

`artifacts/superlender_hgb_pipeline.joblib`

and the configured production metadata:

`artifacts/model_metadata.json`

The production operating threshold is 23% in the current project evidence.

The Streamlit training-session model is intentionally separate and uses a 50% reference threshold until a candidate-specific threshold is explicitly evaluated and promoted.

## FastAPI

Start from the project root:

```bash
python -m uvicorn api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Endpoints:

- `GET /`
- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict/batch`

## Production integration verification

After placing the real project `data/`, `artifacts/`, and current `src/feature_engineer.py` in place:

```bash
python scripts/verify_production_integration.py
```

That check:

1. loads the real model artifact and metadata,
2. loads all three training datasets,
3. builds one real application-time prediction,
4. builds a five-row real batch prediction,
5. checks the 34-feature production contract,
6. checks that feature engineering preserves one row per current loan.

## Tests

The included API contract tests can be run with:

```bash
pytest -q tests/test_fastapi_api.py
```

These tests isolate the HTTP/service contract with controlled test doubles. The production integration script is the check that uses the real project's model artifact and `FeatureEngineer`.

## Defense narrative

The FastAPI design separates responsibilities:

- `api/schemas.py` validates structured requests and rejects target labels.
- `api/main.py` owns HTTP endpoints and dependency wiring.
- `RiskPredictionService` owns inference orchestration.
- `ModelManager` owns model/artifact loading and metadata.
- `FeatureEngineer` owns application-time feature construction and temporal censoring.

The API does not retrain the model on request. It loads the saved production artifact once per API process and returns the predicted class, Bad-risk score, decision threshold, model name and model version.
