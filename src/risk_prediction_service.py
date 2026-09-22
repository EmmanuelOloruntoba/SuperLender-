from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.feature_engineer import FeatureEngineer
from src.model_manager import ModelManager


class RiskPredictionService:
    """Coordinate application-time feature engineering and model inference."""

    CURRENT_REQUIRED_COLUMNS = {
        "systemloanid",
        "customerid",
        "creationdate",
        "loanamount",
        "totaldue",
        "termdays",
        "loannumber",
        "referredby",
    }

    HISTORY_REQUIRED_COLUMNS = {
        "systemloanid",
        "customerid",
        "creationdate",
        "approveddate",
        "loanamount",
        "totaldue",
        "firstduedate",
        "firstrepaiddate",
    }

    DEMOGRAPHIC_REQUIRED_COLUMNS = {
        "customerid",
    }

    def __init__(
        self,
        feature_engineer: FeatureEngineer,
        model_manager: ModelManager,
    ) -> None:
        self.feature_engineer = feature_engineer
        self.model_manager = model_manager

    @staticmethod
    def _records_to_frame(
        records: Sequence[Mapping[str, Any]] | pd.DataFrame,
        empty_columns: Sequence[str],
    ) -> pd.DataFrame:
        if isinstance(records, pd.DataFrame):
            frame = records.copy()
        else:
            frame = pd.DataFrame(list(records))

        if frame.empty:
            return pd.DataFrame(columns=list(empty_columns))
        return frame

    @staticmethod
    def _validate_required_columns(
        frame: pd.DataFrame,
        required: set[str],
        label: str,
    ) -> None:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(
                f"{label} is missing required columns: {missing}"
            )

    def _validate_current_loans(self, current: pd.DataFrame) -> None:
        self._validate_required_columns(
            current,
            self.CURRENT_REQUIRED_COLUMNS,
            "Current-loan data",
        )

        if current.empty:
            raise ValueError("At least one current-loan application is required.")

        if current["systemloanid"].isna().any():
            raise ValueError("Current-loan systemloanid cannot be missing.")

        if current["customerid"].isna().any():
            raise ValueError("Current-loan customerid cannot be missing.")

        if current["systemloanid"].duplicated().any():
            raise ValueError("Current-loan systemloanid must be unique.")

        if "good_bad_flag" in current.columns:
            raise ValueError(
                "good_bad_flag must not be supplied to the prediction workflow."
            )

        for column in ["loanamount", "totaldue", "termdays", "loannumber"]:
            values = pd.to_numeric(current[column], errors="coerce")
            if values.isna().any():
                raise ValueError(
                    f"Current-loan {column} contains non-numeric or missing values."
                )
            if (values <= 0).any():
                raise ValueError(
                    f"Current-loan {column} must contain only positive values."
                )

        loanamount = pd.to_numeric(current["loanamount"], errors="coerce")
        totaldue = pd.to_numeric(current["totaldue"], errors="coerce")
        if (loanamount > totaldue).any():
            raise ValueError("Current-loan loanamount must not exceed totaldue.")

        parsed_creation = pd.to_datetime(
            current["creationdate"],
            errors="coerce",
        )
        if parsed_creation.isna().any():
            raise ValueError("Current-loan creationdate contains invalid values.")

    def _validate_history(self, history: pd.DataFrame) -> None:
        if history.empty:
            return

        self._validate_required_columns(
            history,
            self.HISTORY_REQUIRED_COLUMNS,
            "Historical-loan data",
        )

        if history["customerid"].isna().any():
            raise ValueError("Historical-loan customerid cannot be missing.")

        if history["systemloanid"].duplicated().any():
            raise ValueError("Historical-loan systemloanid must be unique.")

        for column in [
            "creationdate",
            "approveddate",
            "firstduedate",
            "firstrepaiddate",
        ]:
            parsed = pd.to_datetime(history[column], errors="coerce")
            if parsed.isna().any():
                raise ValueError(
                    f"Historical-loan {column} contains invalid values."
                )

    def _validate_demographics(self, demographics: pd.DataFrame) -> None:
        if demographics.empty:
            return

        self._validate_required_columns(
            demographics,
            self.DEMOGRAPHIC_REQUIRED_COLUMNS,
            "Demographic data",
        )

        if demographics["customerid"].isna().any():
            raise ValueError("Demographic customerid cannot be missing.")

    def _build_input_frames(
        self,
        current_loans: pd.DataFrame | Sequence[Mapping[str, Any]],
        previous_loans: pd.DataFrame | Sequence[Mapping[str, Any]],
        demographics: pd.DataFrame | Sequence[Mapping[str, Any]],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        current = self._records_to_frame(
            current_loans,
            empty_columns=sorted(self.CURRENT_REQUIRED_COLUMNS),
        )
        history = self._records_to_frame(
            previous_loans,
            empty_columns=sorted(self.HISTORY_REQUIRED_COLUMNS),
        )
        demo = self._records_to_frame(
            demographics,
            empty_columns=[
                "customerid",
                "birthdate",
                "bank_account_type",
                "employment_status_clients",
                "bank_name_clients",
            ],
        )

        self._validate_current_loans(current)
        self._validate_history(history)
        self._validate_demographics(demo)

        return current, history, demo

    def _expected_features(self) -> list[str]:
        model_features = getattr(
            self.feature_engineer,
            "MODEL_FEATURES",
            None,
        )
        if not model_features:
            raise RuntimeError(
                "FeatureEngineer does not expose MODEL_FEATURES, so the "
                "production feature contract cannot be determined."
            )
        return list(model_features)

    def _predict_engineered(
        self,
        engineered: pd.DataFrame,
    ) -> pd.DataFrame:
        feature_names = self._expected_features()

        missing = [
            feature
            for feature in feature_names
            if feature not in engineered.columns
        ]
        if missing:
            raise RuntimeError(
                "Engineered data is missing required production features: "
                + ", ".join(missing)
            )

        model_input = engineered[feature_names].copy()
        probabilities = self.model_manager.predict_proba(model_input)
        bad_index = self.model_manager.bad_class_index()

        if probabilities.shape[1] <= bad_index:
            raise RuntimeError(
                "The production model returned fewer probability columns than expected."
            )

        bad_scores = probabilities[:, bad_index]
        threshold = self.model_manager.decision_threshold()
        predicted_classes = np.where(
            bad_scores >= threshold,
            "Bad",
            "Good",
        )

        if len(predicted_classes) != len(engineered):
            raise RuntimeError(
                "The model did not return exactly one prediction per application."
            )

        result = engineered[
            ["systemloanid", "customerid"]
        ].copy()
        result["predicted_class"] = predicted_classes
        result["bad_risk_score"] = bad_scores.astype(float)
        result["decision_threshold"] = threshold
        result["model_name"] = self.model_manager.model_name()
        result["model_version"] = self.model_manager.model_version()
        result["risk_event"] = self.model_manager.risk_event()

        if result["systemloanid"].duplicated().any():
            raise RuntimeError(
                "Prediction results contain duplicate current-loan IDs."
            )

        return result

    def predict_single(
        self,
        current_loan: Mapping[str, Any] | pd.Series,
        previous_loans: pd.DataFrame | Sequence[Mapping[str, Any]],
        demographics: pd.DataFrame | Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Score exactly one current application."""
        if isinstance(current_loan, pd.Series):
            current_payload = current_loan.to_dict()
        else:
            current_payload = dict(current_loan)

        current, history, demo = self._build_input_frames(
            [current_payload],
            previous_loans,
            demographics,
        )

        engineered = self.feature_engineer.build_features(
            current_loans=current,
            previous_loans=history,
            demographics=demo,
        )

        if len(engineered) != 1:
            raise RuntimeError(
                "Feature engineering must return exactly one row for a single prediction."
            )

        result = self._predict_engineered(engineered)
        return result.iloc[0].to_dict()

    def predict_batch(
        self,
        current_loans: pd.DataFrame | Sequence[Mapping[str, Any]],
        previous_loans: pd.DataFrame | Sequence[Mapping[str, Any]],
        demographics: pd.DataFrame | Sequence[Mapping[str, Any]],
    ) -> pd.DataFrame:
        """Score one or more current applications."""
        current, history, demo = self._build_input_frames(
            current_loans,
            previous_loans,
            demographics,
        )

        engineered = self.feature_engineer.build_features(
            current_loans=current,
            previous_loans=history,
            demographics=demo,
        )

        if len(engineered) != len(current):
            raise RuntimeError(
                "Feature engineering changed the number of current-loan rows."
            )

        return self._predict_engineered(engineered)
