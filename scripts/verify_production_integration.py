from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineer import FeatureEngineer
from src.model_manager import ModelManager
from src.risk_prediction_service import RiskPredictionService

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "superlender_hgb_pipeline.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"


def main() -> int:
    print("SuperLender production integration verification")
    print("=" * 52)

    required_files = [
        DATA_DIR / "trainperf.csv",
        DATA_DIR / "trainprevloans.csv",
        DATA_DIR / "traindemographics.csv",
        MODEL_PATH,
        METADATA_PATH,
    ]

    missing = [path for path in required_files if not path.exists()]
    if missing:
        print("\nMISSING REQUIRED FILES:")
        for path in missing:
            print(f"  - {path}")
        print("\nPlace your existing production project files in this project's data/ and artifacts/ directories.")
        return 2

    performance = pd.read_csv(DATA_DIR / "trainperf.csv")
    previous_loans = pd.read_csv(DATA_DIR / "trainprevloans.csv")
    demographics = pd.read_csv(DATA_DIR / "traindemographics.csv")

    print(f"Current applications: {len(performance):,}")
    print(f"Historical loans:     {len(previous_loans):,}")
    print(f"Demographic records:  {len(demographics):,}")

    model_manager = ModelManager(MODEL_PATH, METADATA_PATH)
    feature_engineer = FeatureEngineer()
    service = RiskPredictionService(feature_engineer, model_manager)

    print("\nModel metadata:")
    print(json.dumps(model_manager.info(), indent=2, default=str))

    # Use one real current application from trainperf. The observed target is
    # intentionally removed before it enters the prediction service.
    current = performance.iloc[0].drop(labels=["good_bad_flag"]).to_dict()
    customer_id = current["customerid"]

    history = previous_loans[
        previous_loans["customerid"] == customer_id
    ].copy()
    demo = demographics[
        demographics["customerid"] == customer_id
    ].copy()

    prediction = service.predict_single(
        current_loan=current,
        previous_loans=history,
        demographics=demo,
    )

    required = {
        "systemloanid",
        "customerid",
        "predicted_class",
        "bad_risk_score",
        "decision_threshold",
        "model_name",
        "model_version",
        "risk_event",
    }
    missing_response = required - prediction.keys()
    if missing_response:
        raise RuntimeError(f"Prediction response is missing: {sorted(missing_response)}")

    if not 0 <= float(prediction["bad_risk_score"]) <= 1:
        raise RuntimeError("bad_risk_score is outside [0, 1].")

    print("\nSingle prediction: PASS")
    print(json.dumps(prediction, indent=2, default=str))

    batch = performance.iloc[:5].drop(columns=["good_bad_flag"]).copy()
    batch_result = service.predict_batch(
        current_loans=batch,
        previous_loans=previous_loans,
        demographics=demographics,
    )

    if len(batch_result) != len(batch):
        raise RuntimeError(
            f"Batch returned {len(batch_result)} rows for {len(batch)} applications."
        )

    if batch_result["systemloanid"].duplicated().any():
        raise RuntimeError("Batch output contains duplicate systemloanid values.")

    print("Batch prediction:  PASS")
    print(batch_result[[
        "systemloanid",
        "predicted_class",
        "bad_risk_score",
        "decision_threshold",
        "model_name",
        "model_version",
    ]].to_string(index=False))

    # Confirm the feature engineer preserves the exact model feature contract.
    engineered = feature_engineer.build_features(
        current_loans=performance.iloc[:5].copy(),
        previous_loans=previous_loans,
        demographics=demographics,
    )
    expected = list(FeatureEngineer.MODEL_FEATURES)
    missing_features = [f for f in expected if f not in engineered.columns]
    if missing_features:
        raise RuntimeError(f"Feature contract missing: {missing_features}")
    if len(engineered) != 5:
        raise RuntimeError("Feature engineering changed current-loan row count.")

    print("Feature contract:  PASS")
    print(f"Model features:    {len(expected)}")
    print("\nINTEGRATION VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
