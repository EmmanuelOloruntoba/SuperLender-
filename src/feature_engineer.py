from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Build the 34 application-time features expected by the
    trained SuperLender model.

    Inputs:
        - current_loans: current applications being scored
        - previous_loans: historical loans
        - demographics: customer demographic information

    Output:
        One row per current loan containing:
            - systemloanid
            - customerid
            - 34 model features

    Temporal rule:
        Historical information is only used when the historical
        loan was approved before the current application's
        creation time.

        Repayment information is only used when the repayment
        occurred before the current application's creation time.
    """

    # ============================================================
    # MODEL FEATURE CONTRACT
    # ============================================================

    NUMERIC_FEATURES = [
        "loanamount",
        "loan_cost",
        "loan_cost_rate",
        "loannumber",
        "creation_hour",
        "historical_loan_count",
        "history_duration_days",
        "days_since_last_loan",
        "avg_historical_loan_amount",
        "max_historical_loan_amount",
        "total_historical_loan_amount",
        "std_historical_loan_amount",
        "avg_historical_totaldue",
        "max_historical_totaldue",
        "observed_repayment_count",
        "observed_early_repayment_count",
        "observed_on_time_repayment_count",
        "observed_late_repayment_count",
        "observed_late_repayment_rate",
        "observed_early_repayment_rate",
        "observed_on_time_repayment_rate",
        "avg_observed_repayment_delay_days",
        "avg_observed_late_days",
        "age_at_application",
        "referredby_missing",
        "has_historical_history",
        "has_observed_repayment",
        "has_observed_late_repayment",
        "age_missing",
    ]

    CATEGORICAL_FEATURES = [
        "termdays",
        "creation_day_of_week",
        "bank_account_type",
        "employment_status_clients",
        "bank_name_clients",
    ]

    MODEL_FEATURES = (
        NUMERIC_FEATURES +
        CATEGORICAL_FEATURES
    )

    DEMOGRAPHIC_COLUMNS = [
        "customerid",
        "birthdate",
        "bank_account_type",
        "employment_status_clients",
        "bank_name_clients",
    ]

    # Historical columns actually required by our feature logic.
    HISTORICAL_REQUIRED_COLUMNS = [
        "systemloanid",
        "customerid",
        "creationdate",
        "approveddate",
        "firstduedate",
        "firstrepaiddate",
        "loanamount",
        "totaldue",
    ]

    CURRENT_REQUIRED_COLUMNS = [
        "systemloanid",
        "customerid",
        "creationdate",
        "loanamount",
        "totaldue",
        "termdays",
        "loannumber",
        "referredby",
    ]

    HISTORICAL_DATE_COLUMNS = [
        "creationdate",
        "approveddate",
        "firstduedate",
        "firstrepaiddate",
    ]

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(self) -> None:
        pass

    # ============================================================
    # VALIDATION HELPERS
    # ============================================================

    @classmethod
    def _validate_current_loans(
        cls,
        current_loans: pd.DataFrame
    ) -> None:
        """Validate the minimum current-loan input contract."""

        missing = set(cls.CURRENT_REQUIRED_COLUMNS) - set(
            current_loans.columns
        )

        if missing:
            raise ValueError(
                "Current-loan data is missing required columns: "
                f"{sorted(missing)}"
            )

        if current_loans["systemloanid"].isna().any():
            raise ValueError(
                "Current-loan systemloanid cannot be missing."
            )

        if current_loans["systemloanid"].duplicated().any():
            raise ValueError(
                "Current-loan systemloanid must be unique."
            )

        if current_loans["customerid"].isna().any():
            raise ValueError(
                "Current-loan customerid cannot be missing."
            )

        if current_loans["creationdate"].isna().any():
            raise ValueError(
                "Current-loan creationdate cannot be missing."
            )

    @classmethod
    def _validate_previous_loans(
        cls,
        previous_loans: pd.DataFrame
    ) -> None:
        """Validate the historical-loan input contract."""

        missing = set(cls.HISTORICAL_REQUIRED_COLUMNS) - set(
            previous_loans.columns
        )

        if missing:
            raise ValueError(
                "Historical-loan data is missing required columns: "
                f"{sorted(missing)}"
            )

        if previous_loans["systemloanid"].isna().any():
            raise ValueError(
                "Historical systemloanid cannot be missing."
            )

        if previous_loans["systemloanid"].duplicated().any():
            raise ValueError(
                "Historical systemloanid must be unique."
            )

        if previous_loans["customerid"].isna().any():
            raise ValueError(
                "Historical customerid cannot be missing."
            )

    @staticmethod
    def _prepare_demographics(
        demographics: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Clean demographics.

        Exact duplicate rows are removed, matching the notebook
        preprocessing.

        Missing optional demographic columns are retained as
        missing and handled later by the feature logic/model
        preprocessing.
        """

        if "customerid" not in demographics.columns:
            raise ValueError(
                "Demographic data must contain customerid."
            )

        # Reindex creates any missing expected demographic
        # columns as NaN without changing the intended schema.
        cleaned = demographics.reindex(
            columns=FeatureEngineer.DEMOGRAPHIC_COLUMNS
        ).copy()

        # Match notebook behavior:
        # exact duplicate demographic rows are removed.
        cleaned = cleaned.drop_duplicates()

        # After exact duplicate removal, customerid must be unique.
        if cleaned["customerid"].duplicated().any():
            duplicate_ids = (
                cleaned.loc[
                    cleaned["customerid"].duplicated(keep=False),
                    "customerid"
                ]
                .unique()
                .tolist()
            )

            raise ValueError(
                "Demographics contains conflicting duplicate "
                "customer IDs. Example IDs: "
                f"{duplicate_ids[:10]}"
            )

        cleaned["birthdate"] = pd.to_datetime(
            cleaned["birthdate"],
            errors="coerce"
        )

        return cleaned

    # ============================================================
    # EMPTY HISTORICAL FEATURE FRAME
    # ============================================================

    @staticmethod
    def _empty_historical_features() -> pd.DataFrame:
        """Return the expected historical-feature schema."""

        columns = [
            "systemloanid",
            "customerid",
            "historical_loan_count",
            "history_duration_days",
            "days_since_last_loan",
            "avg_historical_loan_amount",
            "max_historical_loan_amount",
            "total_historical_loan_amount",
            "std_historical_loan_amount",
            "avg_historical_totaldue",
            "max_historical_totaldue",
            "observed_repayment_count",
            "observed_early_repayment_count",
            "observed_on_time_repayment_count",
            "observed_late_repayment_count",
            "observed_late_repayment_rate",
            "observed_early_repayment_rate",
            "observed_on_time_repayment_rate",
            "avg_observed_repayment_delay_days",
            "avg_observed_late_days",
        ]

        return pd.DataFrame(columns=columns)

    # ============================================================
    # MAIN FEATURE ENGINEERING
    # ============================================================

    def build_features(
        self,
        current_loans: pd.DataFrame,
        previous_loans: pd.DataFrame,
        demographics: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build exactly one feature row per current loan.
        """

        # --------------------------------------------------------
        # COPY INPUTS
        # --------------------------------------------------------

        current = current_loans.copy()
        history = previous_loans.copy()
        demographics_clean = self._prepare_demographics(
            demographics
        )

        # --------------------------------------------------------
        # VALIDATE INPUTS
        # --------------------------------------------------------

        self._validate_current_loans(current)
        self._validate_previous_loans(history)

        # --------------------------------------------------------
        # DATE CONVERSION
        # --------------------------------------------------------

        current["creationdate"] = pd.to_datetime(
            current["creationdate"],
            errors="raise"
        )

        for column in self.HISTORICAL_DATE_COLUMNS:
            history[column] = pd.to_datetime(
                history[column],
                errors="raise"
            )

        # --------------------------------------------------------
        # SAFETY CHECK:
        # CURRENT LOAN MUST NOT ALSO APPEAR IN HISTORY
        # --------------------------------------------------------

        overlapping_ids = set(
            current["systemloanid"]
        ).intersection(
            history["systemloanid"]
        )

        if overlapping_ids:
            example_ids = list(overlapping_ids)[:10]

            raise ValueError(
                "Potential target-loan leakage detected: "
                "some current systemloanid values also appear "
                "in previous_loans. Example IDs: "
                f"{example_ids}"
            )

        # ========================================================
        # CURRENT-LOAN FEATURES
        # ========================================================

        current["loan_cost"] = (
            current["totaldue"] -
            current["loanamount"]
        )

        current["loan_cost_rate"] = (
            current["loan_cost"] /
            current["loanamount"].replace(0, np.nan)
        )

        current["creation_hour"] = (
            current["creationdate"].dt.hour
        )

        current["creation_day_of_week"] = (
            current["creationdate"].dt.dayofweek
        )

        current["referredby_missing"] = (
            current["referredby"]
            .isna()
            .astype(int)
        )

        # ========================================================
        # DEMOGRAPHICS
        # ========================================================

        # The current application may contain multiple loans for
        # the same customer in a production batch, therefore the
        # correct relationship is many current rows : one
        # demographic row.
        current = current.merge(
            demographics_clean,
            on="customerid",
            how="left",
            validate="many_to_one"
        )

        current["age_at_application"] = (
            (
                current["creationdate"] -
                current["birthdate"]
            ).dt.days / 365.25
        )

        current["age_missing"] = (
            current["age_at_application"]
            .isna()
            .astype(int)
        )

        # ========================================================
        # PREPARE CURRENT-LOAN IDENTIFIERS FOR HISTORY JOIN
        # ========================================================

        current_for_history = (
            current[
                [
                    "systemloanid",
                    "customerid",
                    "creationdate",
                ]
            ]
            .rename(
                columns={
                    "systemloanid": "current_systemloanid",
                    "creationdate": "current_creationdate",
                }
            )
        )

        # ========================================================
        # JOIN HISTORY TO CURRENT APPLICATIONS
        # ========================================================

        history = history.merge(
            current_for_history,
            on="customerid",
            how="inner",
            validate="many_to_many"
        )

        # ========================================================
        # TEMPORAL LEAKAGE CONTROL
        # ========================================================

        # A historical loan is eligible only if it was approved
        # before the current application's creation time.
        history = history[
            history["approveddate"] <
            history["current_creationdate"]
        ].copy()

        # ========================================================
        # HISTORICAL FEATURE AGGREGATION
        # ========================================================

        if history.empty:

            historical_features = (
                self._empty_historical_features()
            )

        else:

            history_keys = [
                "current_systemloanid",
                "customerid"
            ]

            # ----------------------------------------------------
            # HISTORY SUMMARY
            # ----------------------------------------------------

            history_summary = (
                history
                .groupby(
                    history_keys,
                    as_index=False
                )
                .agg(
                    historical_loan_count=(
                        "systemloanid",
                        "size"
                    ),
                    first_historical_loan_date=(
                        "creationdate",
                        "min"
                    ),
                    last_historical_approval_date=(
                        "approveddate",
                        "max"
                    ),
                    current_creationdate=(
                        "current_creationdate",
                        "first"
                    ),
                )
            )

            history_summary["history_duration_days"] = (
                (
                    history_summary["current_creationdate"] -
                    history_summary["first_historical_loan_date"]
                )
                .dt.total_seconds()
                / (24 * 60 * 60)
            )

            history_summary["days_since_last_loan"] = (
                (
                    history_summary["current_creationdate"] -
                    history_summary["last_historical_approval_date"]
                )
                .dt.total_seconds()
                / (24 * 60 * 60)
            )

            history_summary = history_summary[
                [
                    "current_systemloanid",
                    "customerid",
                    "historical_loan_count",
                    "history_duration_days",
                    "days_since_last_loan",
                ]
            ]

            # ----------------------------------------------------
            # BORROWING FEATURES
            # ----------------------------------------------------

            borrowing = (
                history
                .groupby(
                    history_keys,
                    as_index=False
                )
                .agg(
                    avg_historical_loan_amount=(
                        "loanamount",
                        "mean"
                    ),
                    max_historical_loan_amount=(
                        "loanamount",
                        "max"
                    ),
                    total_historical_loan_amount=(
                        "loanamount",
                        "sum"
                    ),
                    std_historical_loan_amount=(
                        "loanamount",
                        "std"
                    ),
                    avg_historical_totaldue=(
                        "totaldue",
                        "mean"
                    ),
                    max_historical_totaldue=(
                        "totaldue",
                        "max"
                    ),
                )
            )

            # ----------------------------------------------------
            # REPAYMENT TIMING
            # ----------------------------------------------------

            history["repayment_delay_days"] = (
                (
                    history["firstrepaiddate"] -
                    history["firstduedate"]
                )
                .dt.total_seconds()
                / (24 * 60 * 60)
            )

            # A repayment is observable only when it occurred
            # before the current application's creation time.
            history["repayment_observed"] = (
                history["firstrepaiddate"].notna()
                &
                (
                    history["firstrepaiddate"] <
                    history["current_creationdate"]
                )
            )

            observed = history["repayment_observed"]

            history["early_flag"] = (
                observed
                &
                (history["repayment_delay_days"] < 0)
            )

            history["on_time_flag"] = (
                observed
                &
                (history["repayment_delay_days"] == 0)
            )

            history["late_flag"] = (
                observed
                &
                (history["repayment_delay_days"] > 0)
            )

            # Values outside the relevant condition become NaN,
            # so mean() naturally operates only over applicable
            # observations.
            history["observed_delay"] = (
                history["repayment_delay_days"]
                .where(observed)
            )

            history["observed_late_days"] = (
                history["repayment_delay_days"]
                .where(history["late_flag"])
            )

            # ----------------------------------------------------
            # REPAYMENT FEATURES
            # ----------------------------------------------------

            repayment = (
                history
                .groupby(
                    history_keys,
                    as_index=False
                )
                .agg(
                    observed_repayment_count=(
                        "repayment_observed",
                        "sum"
                    ),
                    observed_early_repayment_count=(
                        "early_flag",
                        "sum"
                    ),
                    observed_on_time_repayment_count=(
                        "on_time_flag",
                        "sum"
                    ),
                    observed_late_repayment_count=(
                        "late_flag",
                        "sum"
                    ),
                    avg_observed_repayment_delay_days=(
                        "observed_delay",
                        "mean"
                    ),
                    avg_observed_late_days=(
                        "observed_late_days",
                        "mean"
                    ),
                )
            )

            # ----------------------------------------------------
            # REPAYMENT RATES
            # ----------------------------------------------------

            repayment["observed_late_repayment_rate"] = (
                repayment["observed_late_repayment_count"] /
                repayment["observed_repayment_count"]
            )

            repayment["observed_early_repayment_rate"] = (
                repayment["observed_early_repayment_count"] /
                repayment["observed_repayment_count"]
            )

            repayment["observed_on_time_repayment_rate"] = (
                repayment["observed_on_time_repayment_count"] /
                repayment["observed_repayment_count"]
            )

            # ----------------------------------------------------
            # COMBINE HISTORICAL FEATURES
            # ----------------------------------------------------

            historical_features = (
                history_summary
                .merge(
                    borrowing,
                    on=history_keys,
                    how="left",
                    validate="one_to_one"
                )
                .merge(
                    repayment,
                    on=history_keys,
                    how="left",
                    validate="one_to_one"
                )
                .rename(
                    columns={
                        "current_systemloanid":
                        "systemloanid"
                    }
                )
            )

        # ========================================================
        # MERGE HISTORICAL FEATURES INTO CURRENT LOANS
        # ========================================================

        current = current.merge(
            historical_features,
            on=[
                "systemloanid",
                "customerid"
            ],
            how="left",
            validate="one_to_one"
        )

        # ========================================================
        # STRUCTURAL MISSINGNESS
        # ========================================================

        # This represents whether historical loan records existed
        # at application time.
        current["has_historical_history"] = (
            current["historical_loan_count"]
            .notna()
            .astype(int)
        )

        # This represents whether at least one historical repayment
        # was actually observable before the current application.
        current["has_observed_repayment"] = (
            current["observed_repayment_count"]
            .fillna(0)
            .gt(0)
            .astype(int)
        )

        # This represents whether at least one observed historical
        # repayment was late.
        current["has_observed_late_repayment"] = (
            current["observed_late_repayment_count"]
            .fillna(0)
            .gt(0)
            .astype(int)
        )

        historical_zero_fill_cols = [
            "historical_loan_count",
            "history_duration_days",
            "days_since_last_loan",
            "avg_historical_loan_amount",
            "max_historical_loan_amount",
            "total_historical_loan_amount",
            "std_historical_loan_amount",
            "avg_historical_totaldue",
            "max_historical_totaldue",
            "observed_repayment_count",
            "observed_early_repayment_count",
            "observed_on_time_repayment_count",
            "observed_late_repayment_count",
            "observed_late_repayment_rate",
            "observed_early_repayment_rate",
            "observed_on_time_repayment_rate",
            "avg_observed_repayment_delay_days",
            "avg_observed_late_days",
        ]

        current[historical_zero_fill_cols] = (
            current[
                historical_zero_fill_cols
            ].fillna(0)
        )

        # ========================================================
        # CATEGORICAL STRUCTURAL MISSINGNESS
        # ========================================================

        categorical_unknown_cols = [
            "bank_account_type",
            "employment_status_clients",
            "bank_name_clients",
        ]

        for column in categorical_unknown_cols:

            current[column] = (
                current[column]
                .fillna("Unknown")
            )

        # ========================================================
        # FINAL CONTRACT
        # ========================================================

        result = current[
            [
                "systemloanid",
                "customerid",
                *self.MODEL_FEATURES,
            ]
        ].copy()

        # ========================================================
        # FINAL VALIDATION
        # ========================================================

        if len(result) != len(current_loans):
            raise ValueError(
                "Feature engineering changed the number of "
                "current-loan rows. "
                f"Input rows={len(current_loans)}, "
                f"output rows={len(result)}."
            )

        if result["systemloanid"].duplicated().any():
            raise ValueError(
                "Feature engineering produced duplicate "
                "current systemloanid values."
            )

        if result["customerid"].isna().any():
            raise ValueError(
                "Feature engineering produced missing customerid."
            )

        actual_model_features = [
            column
            for column in result.columns
            if column not in [
                "systemloanid",
                "customerid"
            ]
        ]

        if actual_model_features != self.MODEL_FEATURES:
            raise ValueError(
                "Final model feature schema does not match "
                "the expected 34-feature contract."
            )

        return result