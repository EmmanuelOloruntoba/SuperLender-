from __future__ import annotations

import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from textwrap import dedent

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
)
# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

MODEL_PATH = (
    ARTIFACT_DIR /
    "superlender_hgb_pipeline.joblib"
)

METADATA_PATH = (
    ARTIFACT_DIR /
    "model_metadata.json"
)

IMPORTANCE_PATH = (
    ARTIFACT_DIR /
    "permutation_importance.csv"
)

OPERATING_POINTS_PATH = (
    ARTIFACT_DIR /
    "final_operating_points.csv"
)

EVIDENCE_PATH = (
    ARTIFACT_DIR /
    "model_evidence.json"
)

FAIRNESS_PATH = (ARTIFACT_DIR / "fairness_audit.csv")

# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.feature_engineer import FeatureEngineer
from src.model_manager import ModelManager
from src.risk_prediction_service import RiskPredictionService

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SuperLender | Credit Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)



# ============================================================
# GLOBAL STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Main container ---------- */

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        padding-top: 1.5rem;
    }


    /* ---------- Main title ---------- */

    .brand-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }

    .brand-subtitle {
        font-size: 1rem;
        opacity: 0.70;
        margin-bottom: 2rem;
    }


    /* ---------- Metric cards ---------- */

    .metric-card {
        padding: 1.15rem 1.25rem;
        border-radius: 14px;
        border: 1px solid rgba(128, 128, 128, 0.20);
        background: rgba(128, 128, 128, 0.05);
        min-height: 120px;
    }

    .metric-label {
        font-size: 0.82rem;
        opacity: 0.68;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 750;
        line-height: 1.1;
    }

    .metric-caption {
        font-size: 0.78rem;
        opacity: 0.62;
        margin-top: 0.4rem;
    }


    /* ---------- Section headers ---------- */

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.25rem;
    }

    .section-description {
        font-size: 0.9rem;
        opacity: 0.68;
        margin-bottom: 1rem;
    }


    /* ---------- Risk score ---------- */

    .risk-score {
        font-size: 3.4rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.35rem 0;
    }

    .risk-label {
        font-size: 0.85rem;
        opacity: 0.68;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }


    /* ---------- Status box ---------- */

    .status-box {
        padding: 1rem 1.15rem;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.20);
        margin-top: 1rem;
    }

    .status-title {
        font-size: 0.8rem;
        opacity: 0.68;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    .status-value {
        font-size: 1.5rem;
        font-weight: 750;
        margin-top: 0.2rem;
    }


    /* ---------- Footer ---------- */

    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(128, 128, 128, 0.20);
        font-size: 0.78rem;
        opacity: 0.58;
    }

    /*  ============================================================
        RISK SCORE VISUAL
        ============================================================ */

    .risk-gauge-wrapper {
        margin-top: 2.6rem;          /* room for the label above the bar */
        margin-bottom: 1.3rem;
    }

    .risk-gauge {
        position: relative;
        width: 100%;
        height: 14px;
        border-radius: 999px;
    }

    .risk-threshold {
        position: absolute;
        top: -8px;
        height: 30px;                /* 14px bar + 8px above + 8px below */
        transform: translateX(-50%);
        z-index: 1;
    }

    .risk-threshold-line {
        width: 2px;
        height: 100%;
        background: currentColor;
        box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.35);   /* keeps it visible on the pastel bar */
    }

    .risk-threshold-label {
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        margin-bottom: 4px;
        font-size: 0.72rem;
        font-weight: 700;
        white-space: nowrap;
    }

    .risk-marker {
        position: absolute;
        top: 50%;
        transform: translate(-50%, -50%);
        z-index: 2;                  /* marker always above the threshold line */
    }

    .risk-marker-dot {
        width: 19px;
        height: 19px;
        border-radius: 50%;
        border: 3px solid white;
        box-shadow: 0 1px 6px rgba(0, 0, 0, 0.35);
    }

    .risk-gauge-labels {
        display: flex;
        justify-content: space-between;
        margin-top: 0.5rem;
        font-size: 0.72rem;
        opacity: 0.65;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# REUSABLE UI HELPERS
# ============================================================

def metric_card(
    label: str,
    value: str,
    caption: str = "",
) -> None:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(
    title: str,
    description: str = "",
) -> None:

    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )

    if description:
        st.markdown(
            f'<div class="section-description">'
            f'{description}'
            f'</div>',
            unsafe_allow_html=True,
        )

def get_dataset_metadata(
    df: pd.DataFrame,
    key_columns: list[str] | None = None,
    date_columns: list[str] | None = None,
) -> dict:
    """Build a compact quality summary for one project dataset."""

    metadata = {
        "rows": len(df),
        "columns": len(df.columns),
        "exact_duplicate_rows": int(df.duplicated().sum()),
    }

    if key_columns:
        for key in key_columns:
            if key in df.columns:
                duplicated_ids = df.loc[
                    df[key].duplicated(keep=False),
                    key,
                ].dropna()

                metadata[f"{key}_unique"] = int(
                    df[key].nunique(dropna=True)
                )

                metadata[f"{key}_duplicate_ids"] = int(
                    duplicated_ids.nunique()
                )

    if date_columns:
        date_ranges = {}

        for column in date_columns:
            if column in df.columns:
                parsed = pd.to_datetime(
                    df[column],
                    errors="coerce",
                )

                valid = parsed.dropna()

                if not valid.empty:
                    date_ranges[column] = (
                        valid.min(),
                        valid.max(),
                    )

        metadata["date_ranges"] = date_ranges

    return metadata


def build_working_dataset_preview(
    performance: pd.DataFrame,
    previous_loans: pd.DataFrame,
    demographics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Build a current-loan-level preview that demonstrates the
    relational merge without creating duplicate current rows.
    """

    current = performance.copy()

    current["creationdate"] = pd.to_datetime(
        current["creationdate"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Deduplicate demographics at customer level
    # --------------------------------------------------------

    demographics_dedup = (
        demographics
        .drop_duplicates()
        .drop_duplicates(
            subset=["customerid"],
            keep="first",
        )
        .copy()
    )

    before_demo = len(current)

    current_with_demo = current.merge(
        demographics_dedup,
        on="customerid",
        how="left",
        indicator=True,
    )

    demo_match_counts = (
        current_with_demo["_merge"]
        .value_counts()
        .to_dict()
    )

    current_with_demo = current_with_demo.drop(
        columns=["_merge"]
    )

    # --------------------------------------------------------
    # Aggregate historical loans to customer level
    # --------------------------------------------------------

    history_counts = (
        previous_loans
        .groupby("customerid")
        .size()
        .rename("historical_loan_count")
        .reset_index()
    )

    working = current_with_demo.merge(
        history_counts,
        on="customerid",
        how="left",
    )

    working["historical_loan_count"] = (
        working["historical_loan_count"]
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Audit the result
    # --------------------------------------------------------

    audit = {
        "current_rows_before_merge": before_demo,
        "rows_after_demographics_merge": len(current_with_demo),
        "rows_after_history_aggregation": len(working),
        "current_unique_systemloanid": int(
            performance["systemloanid"].nunique()
        ),
        "current_unique_customerid": int(
            performance["customerid"].nunique()
        ),
        "demographic_matched": int(
            demo_match_counts.get("both", 0)
        ),
        "demographic_unmatched": int(
            demo_match_counts.get("left_only", 0)
        ),
        "current_with_history": int(
            (working["historical_loan_count"] > 0).sum()
        ),
        "current_without_history": int(
            (working["historical_loan_count"] == 0).sum()
        ),
        "final_duplicate_systemloanid": int(
            working["systemloanid"].duplicated().sum()
        ),
    }

    return working, audit


def display_date_ranges(
    df: pd.DataFrame,
    date_columns: list[str],
) -> None:
    """Display valid date ranges for a selected dataset."""

    rows = []

    for column in date_columns:

        if column not in df.columns:
            continue

        parsed = pd.to_datetime(
            df[column],
            errors="coerce",
        )

        valid = parsed.dropna()

        if valid.empty:
            rows.append(
                {
                    "Column": column,
                    "Earliest": "No valid dates",
                    "Latest": "No valid dates",
                }
            )
        else:
            rows.append(
                {
                    "Column": column,
                    "Earliest": valid.min().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "Latest": valid.max().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }
            )

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No date columns available for this dataset.")


def display_missingness(df: pd.DataFrame) -> None:
    """Display column-level missingness."""

    missing = (
        df.isna()
        .sum()
        .rename("Missing")
        .to_frame()
    )

    missing["Missing_%"] = (
        missing["Missing"] / len(df) * 100
    ).round(2)

    missing = (
        missing
        .sort_values(
            ["Missing", "Missing_%"],
            ascending=False,
        )
    )

    missing = missing[
        missing["Missing"] > 0
    ]

    if missing.empty:
        st.success("No missing values detected.")
        return

    st.dataframe(
        missing,
        use_container_width=True,
    )


def display_key_quality(
    df: pd.DataFrame,
    key_columns: list[str],
) -> None:
    """Display uniqueness and duplicate-key diagnostics."""

    rows = []

    for column in key_columns:

        if column not in df.columns:
            continue

        duplicate_rows = df[column].duplicated(
            keep=False
        )

        duplicate_ids = (
            df.loc[duplicate_rows, column]
            .dropna()
            .nunique()
        )

        rows.append(
            {
                "Column": column,
                "Unique values": int(
                    df[column].nunique(dropna=True)
                ),
                "Missing": int(
                    df[column].isna().sum()
                ),
                "Rows involved in duplicate keys": int(
                    duplicate_rows.sum()
                ),
                "Duplicate key values": int(
                    duplicate_ids
                ),
            }
        )

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

def build_eda_dataset(
    performance: pd.DataFrame,
    previous_loans: pd.DataFrame,
    demographics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the EDA dataset from the canonical production
    feature-engineering pipeline.

    The resulting dataframe contains:
        - current-loan identifiers
        - 34 production model features
        - observed target
        - EDA-only display/helper columns

    Historical features therefore follow the same temporal rules
    used by production inference.
    """

    feature_engineer = FeatureEngineer()

    engineered = feature_engineer.build_features(
        current_loans=performance,
        previous_loans=previous_loans,
        demographics=demographics,
    )

    target = performance[
        [
            "systemloanid",
            "customerid",
            "good_bad_flag",
        ]
    ].copy()

    eda_data = target.merge(
        engineered,
        on=[
            "systemloanid",
            "customerid",
        ],
        how="left",
        validate="one_to_one",
    )

    # ========================================================
    # EDA-ONLY DISPLAY VARIABLES
    # ========================================================

    day_names = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }

    eda_data["creation_day_name"] = (
        eda_data["creation_day_of_week"]
        .map(day_names)
    )

    eda_data["history_depth_band"] = pd.cut(
        eda_data["historical_loan_count"],
        bins=[
            -0.001,
            0,
            2,
            5,
            9,
            float("inf"),
        ],
        labels=[
            "No history",
            "1–2",
            "3–5",
            "6–9",
            "10+",
        ],
    )

    eda_data["late_rate_band"] = pd.cut(
        eda_data["observed_late_repayment_rate"],
        bins=[
            -0.001,
            0,
            0.25,
            0.50,
            0.75,
            1.001,
        ],
        labels=[
            "0%",
            "0–25%",
            "25–50%",
            "50–75%",
            "75–100%",
        ],
    )

    # ========================================================
    # INTEGRITY CHECK
    # ========================================================

    expected_features = FeatureEngineer.MODEL_FEATURES

    missing_features = [
        column
        for column in expected_features
        if column not in eda_data.columns
    ]

    if missing_features:
        raise ValueError(
            "EDA dataset is missing production features: "
            f"{missing_features}"
        )

    if eda_data["systemloanid"].duplicated().any():
        raise ValueError(
            "EDA dataset contains duplicate current loan IDs."
        )

    if len(eda_data) != len(performance):
        raise ValueError(
            "EDA dataset changed the current-loan row count."
        )

    return eda_data


def create_bad_rate_summary(
    df: pd.DataFrame,
    group_column: str,
    minimum_group_size: int = 10,
) -> pd.DataFrame:
    """Calculate observed Bad rate by a grouping variable."""

    summary = (
        df.dropna(subset=[group_column])
        .groupby(group_column, observed=False)
        .agg(
            applications=("good_bad_flag", "size"),
            bad_count=(
                "good_bad_flag",
                lambda x: (x == "Bad").sum(),
            ),
            bad_rate=(
                "good_bad_flag",
                lambda x: (x == "Bad").mean(),
            ),
        )
        .reset_index()
    )

    return summary[
        summary["applications"] >= minimum_group_size
    ].copy()


def add_percentage_labels(
    ax,
    bars,
    values,
) -> None:
    """Add percentage labels above bar-chart bars."""

    for bar, value in zip(bars, values):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.005,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def observation_and_inference(
    summary: pd.DataFrame,
    group_column: str,
    variable_name: str,
) -> None:
    """
    Display evidence-based observation and inference for
    a Bad-rate summary.
    """

    if summary.empty:
        st.info(
            "There are not enough observations to generate "
            "a stable subgroup comparison."
        )
        return

    highest = summary.loc[
        summary["bad_rate"].idxmax()
    ]

    lowest = summary.loc[
        summary["bad_rate"].idxmin()
    ]

    st.markdown("#### Observation")

    st.write(
        f"In this sample, the highest observed Bad rate was "
        f"{highest['bad_rate']:.1%} in "
        f"**{highest[group_column]}** "
        f"(n={int(highest['applications']):,}), while the lowest "
        f"was {lowest['bad_rate']:.1%} in "
        f"**{lowest[group_column]}** "
        f"(n={int(lowest['applications']):,})."
    )

    st.markdown("#### Inference")

    st.write(
        f"The results indicate an association between "
        f"**{variable_name}** and observed Bad outcomes in "
        "this dataset. The analysis does not establish that "
        "the variable causes default, because other customer, "
        "loan or behavioural characteristics may also contribute "
        "to the observed differences."
    )


def figure_download_button(
    fig,
    filename: str,
    label: str = "Download chart",
) -> None:
    """Create a PNG download button for a Matplotlib figure."""

    import io

    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=180,
        bbox_inches="tight",
    )

    buffer.seek(0)

    st.download_button(
        label,
        data=buffer,
        file_name=filename,
        mime="image/png",
    )


def display_risk_result(
    bad_risk_score: float,
    threshold: float,
    predicted_class: str,
) -> None:
    """
    Display the model's Bad-risk score, operating threshold,
    and resulting classification.
    """

    score_percent = bad_risk_score * 100
    threshold_percent = threshold * 100

    score_position = min(
        max(score_percent, 0.0),
        100.0,
    )

    threshold_position = min(
        max(threshold_percent, 0.0),
        100.0,
    )

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    st.markdown(
        '<div class="risk-label">Estimated Bad Risk</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="risk-score">{score_percent:.1f}%</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Score / threshold visual
    # --------------------------------------------------------

    marker_colour = "#c62828" if predicted_class == "Bad" else "#2e7d32"

    bar_background = (
        "linear-gradient(to right, "
        f"#cfe8d5 0%, #cfe8d5 {threshold_position}%, "
        f"#f4cccc {threshold_position}%, #f4cccc 100%)"
    )

    gauge_html = f"""
    <div class="risk-gauge-wrapper">
        <div class="risk-gauge" style="background: {bar_background};">
            <div class="risk-threshold" style="left: {threshold_position}%">
                <div class="risk-threshold-label">Threshold {threshold_percent:.1f}%</div>
                <div class="risk-threshold-line"></div>
            </div>
            <div class="risk-marker" style="left: {score_position}%">
                <div class="risk-marker-dot" style="background: {marker_colour};"></div>
            </div>
        </div>
        <div class="risk-gauge-labels">
            <span>0%</span>
            <span>100%</span>
        </div>
    </div>
    """

    gauge_html = " ".join(
        line.strip() for line in gauge_html.splitlines() if line.strip()
    )

    st.markdown(gauge_html, unsafe_allow_html=True)

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if predicted_class == "Bad":

        st.error(
            f"**Model decision: Bad**  \n"
            f"The estimated Bad risk is above the "
            f"{threshold_percent:.1f}% operating threshold."
        )

    else:

        st.success(
            f"**Model decision: Good**  \n"
            f"The estimated Bad risk is below the "
            f"{threshold_percent:.1f}% operating threshold."
        )

    st.caption(
        "The score is the model's estimated Bad-class risk score. "
        "It should not be interpreted as a perfectly calibrated "
        "probability or a guaranteed lending outcome."
    )


def calculate_iqr_outliers(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Calculate IQR-based outlier counts for selected numerical columns.

    Outliers are flagged statistically for investigation only.
    They are not automatically removed.
    """

    rows = []

    for column in columns:

        if column not in df.columns:
            continue

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (
            (series < lower_bound)
            | (series > upper_bound)
        )

        rows.append(
            {
                "Column": column,
                "Q1": q1,
                "Q3": q3,
                "Lower bound": lower_bound,
                "Upper bound": upper_bound,
                "Outliers": int(outlier_mask.sum()),
                "Outlier %": (
                    outlier_mask.mean() * 100
                ),
            }
        )

    return pd.DataFrame(rows)


def check_current_data_consistency(
    performance: pd.DataFrame,
) -> pd.DataFrame:
    """Check known logical constraints in current applications."""

    df = performance.copy()

    creation = pd.to_datetime(
        df["creationdate"],
        errors="coerce",
    )

    approved = pd.to_datetime(
        df["approveddate"],
        errors="coerce",
    )

    checks = []

    checks.append(
        {
            "Check": "systemloanid uniqueness",
            "Violations": int(
                df["systemloanid"].duplicated().sum()
            ),
            "Expected": 0,
        }
    )

    checks.append(
        {
            "Check": "customerid uniqueness at current-loan grain",
            "Violations": int(
                df["customerid"].duplicated().sum()
            ),
            "Expected": 0,
        }
    )

    checks.append(
        {
            "Check": "loanamount <= totaldue",
            "Violations": int(
                (df["loanamount"] > df["totaldue"]).sum()
            ),
            "Expected": 0,
        }
    )

    checks.append(
        {
            "Check": "approveddate >= creationdate",
            "Violations": int(
                (approved < creation).sum()
            ),
            "Expected": 0,
        }
    )

    checks.append(
        {
            "Check": "loanamount > 0",
            "Violations": int(
                (df["loanamount"] <= 0).sum()
            ),
            "Expected": 0,
        }
    )

    checks.append(
        {
            "Check": "totaldue > 0",
            "Violations": int(
                (df["totaldue"] <= 0).sum()
            ),
            "Expected": 0,
        }
    )

    return pd.DataFrame(checks)


def check_historical_data_consistency(
    previous_loans: pd.DataFrame,
) -> pd.DataFrame:
    """Check known logical constraints in historical loans."""

    df = previous_loans.copy()

    creation = pd.to_datetime(
        df["creationdate"],
        errors="coerce",
    )

    approved = pd.to_datetime(
        df["approveddate"],
        errors="coerce",
    )

    first_due = pd.to_datetime(
        df["firstduedate"],
        errors="coerce",
    )

    first_repaid = pd.to_datetime(
        df["firstrepaiddate"],
        errors="coerce",
    )

    closed = pd.to_datetime(
        df["closeddate"],
        errors="coerce",
    )

    checks = []

    checks.append(
        {
            "Check": "systemloanid uniqueness",
            "Violations": int(
                df["systemloanid"].duplicated().sum()
            ),
            "Expected": 0,
        }
    )

    checks.append(
        {
            "Check": "approveddate >= creationdate",
            "Violations": int(
                (approved < creation).sum()
            ),
            "Expected": 0,
        }
    )

    checks.append(
        {
            "Check": "firstduedate >= approveddate",
            "Violations": int(
                (first_due < approved).sum()
            ),
            "Expected": 0,
        }
    )

    checks.append(
        {
            "Check": "closeddate >= creationdate",
            "Violations": int(
                (closed < creation).sum()
            ),
            "Expected": 0,
        }
    )

    checks.append(
        {
            "Check": "closeddate >= firstrepaiddate",
            "Violations": int(
                (closed < first_repaid).sum()
            ),
            "Expected": 0,
        }
    )

    checks.append(
        {
            "Check": "loanamount <= totaldue",
            "Violations": int(
                (df["loanamount"] > df["totaldue"]).sum()
            ),
            "Expected": 0,
        }
    )

    return pd.DataFrame(checks)


def find_gps_anomalies(
    demographics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify GPS values outside the broad Nigeria-like coordinate
    ranges used during the original diagnostic investigation.

    This is a diagnostic flag, not a geographic truth claim.
    """
    df = demographics.copy()

    latitude = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    longitude = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    suspicious_latitude = (
        latitude.notna()
        & (
            (latitude < 4)
            | (latitude > 14)
        )
    )

    suspicious_longitude = (
        longitude.notna()
        & (
            (longitude < 2)
            | (longitude > 15)
        )
    )

    suspicious = (
        suspicious_latitude
        | suspicious_longitude
    )

    result = df.loc[
        suspicious,
        [
            "customerid",
            "latitude",
            "longitude",
        ],
    ].copy()

    return result


def quality_status(
    violations: int,
) -> str:
    """Return a simple diagnostic status."""

    return "PASS" if violations == 0 else "REVIEW"

# ============================================================
# ANALYTICAL HELPER
# ============================================================

def create_late_rate_analysis(
    previous_loans: pd.DataFrame,
    performance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create an application-level dataset containing each current
    loan's historical late-repayment rate.

    Temporal rule:
        Only historical loans approved before the current
        application's creation time are eligible.

    Repayment-observation rule:
        A historical repayment is only considered observed if
        its first repayment date occurred before the current
        application's creation time.

    Output grain:
        One row per current application.
    """

    # --------------------------------------------------------
    # Copy inputs so the original DataFrames are not modified
    # --------------------------------------------------------

    historical = previous_loans.copy()

    current = performance[
        [
            "systemloanid",
            "customerid",
            "creationdate",
            "good_bad_flag",
        ]
    ].copy()


    # --------------------------------------------------------
    # Convert relevant date columns
    # --------------------------------------------------------

    historical_date_cols = [
        "creationdate",
        "approveddate",
        "firstduedate",
        "firstrepaiddate",
    ]

    for column in historical_date_cols:
        historical[column] = pd.to_datetime(
            historical[column],
            errors="coerce",
        )

    current["creationdate"] = pd.to_datetime(
        current["creationdate"],
        errors="coerce",
    )


    # --------------------------------------------------------
    # Rename current/historical identifiers explicitly
    #
    # This prevents pandas merge suffix problems.
    # --------------------------------------------------------

    current = current.rename(
        columns={
            "systemloanid": "current_systemloanid",
            "creationdate": "current_creationdate",
        }
    )

    historical = historical.rename(
        columns={
            "systemloanid": "historical_systemloanid",
        }
    )


    # --------------------------------------------------------
    # Join historical loans to current applications
    #
    # Relationship:
    #     customerid
    #
    # One customer can have multiple historical loans.
    # Each current application is represented separately.
    # --------------------------------------------------------

    merged = historical.merge(
        current,
        on="customerid",
        how="inner",
        validate="many_to_one",
    )


    # --------------------------------------------------------
    # Temporal availability boundary
    #
    # Historical loan must have been approved before the
    # current application was created.
    # --------------------------------------------------------

    merged = merged[
        merged["approveddate"]
        < merged["current_creationdate"]
    ].copy()


    # --------------------------------------------------------
    # Repayment observation boundary
    #
    # A repayment event is available to the model only if it
    # happened before the current application.
    # --------------------------------------------------------

    merged["repayment_observed"] = (
        merged["firstrepaiddate"]
        < merged["current_creationdate"]
    )


    # Keep only repayment outcomes observable at application time
    merged = merged[
        merged["repayment_observed"]
    ].copy()


    # --------------------------------------------------------
    # Repayment delay
    #
    # Negative = repaid early
    # Zero     = repaid exactly on due date
    # Positive = repaid late
    # --------------------------------------------------------

    merged["repayment_delay_days"] = (
        merged["firstrepaiddate"]
        - merged["firstduedate"]
    ).dt.total_seconds() / 86400


    # --------------------------------------------------------
    # Late-payment indicator
    # --------------------------------------------------------

    merged["late"] = (
        merged["repayment_delay_days"] > 0
    ).astype(int)


    # --------------------------------------------------------
    # Aggregate to current-loan level
    #
    # IMPORTANT:
    # We group by CURRENT loan ID, not the historical loan ID.
    # This gives us one row per prediction.
    # --------------------------------------------------------

    late_rate = (
        merged
        .groupby("current_systemloanid")["late"]
        .mean()
        .rename("historical_late_rate")
        .reset_index()
    )


    # --------------------------------------------------------
    # Merge the feature back onto ALL current applications
    #
    # Left join preserves current applications that have no
    # observable historical repayment information.
    # --------------------------------------------------------

    analysis = current.merge(
        late_rate,
        on="current_systemloanid",
        how="left",
        validate="one_to_one",
    )


    # Restore the original identifier names
    analysis = analysis.rename(
        columns={
            "current_systemloanid": "systemloanid",
            "current_creationdate": "creationdate",
        }
    )


    # --------------------------------------------------------
    # Basic integrity checks
    # --------------------------------------------------------

    if len(analysis) != len(performance):
        raise ValueError(
            "Late-rate analysis changed the number of current "
            "applications."
        )

    if not analysis["systemloanid"].is_unique:
        raise ValueError(
            "Late-rate analysis does not contain one row per "
            "current loan."
        )


    return analysis

def prepare_training_frame(
    performance: pd.DataFrame,
    previous_loans: pd.DataFrame,
    demographics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the canonical 34-feature modelling frame using
    the same FeatureEngineer used by production inference.
    """

    feature_engineer = FeatureEngineer()

    engineered = feature_engineer.build_features(
        current_loans=performance,
        previous_loans=previous_loans,
        demographics=demographics,
    )

    target = performance[
        [
            "systemloanid",
            "customerid",
            "creationdate",
            "good_bad_flag",
        ]
    ].copy()

    target["creationdate"] = pd.to_datetime(
        target["creationdate"],
        errors="raise",
    )

    frame = target.merge(
        engineered,
        on=[
            "systemloanid",
            "customerid",
        ],
        how="left",
        validate="one_to_one",
    )

    return frame.sort_values(
        "creationdate"
    ).reset_index(drop=True)


def make_training_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def chronological_train_validation_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train_end = int(len(frame) * 0.70)
    validation_end = int(len(frame) * 0.85)


    train = frame.iloc[:train_end].copy()
    validation = frame.iloc[train_end:validation_end].copy()
    final_holdout = frame.iloc[validation_end:].copy()

    return train, validation, final_holdout


def evaluate_bad_risk_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:

    y_bad = (
        y == 0
    ).astype(int)

    probabilities = pipeline.predict_proba(X)

    classes = list(
        pipeline.named_steps["model"].classes_
    )

    bad_index = classes.index(0)

    bad_risk = probabilities[
        :,
        bad_index,
    ]

    predicted_bad = (
        bad_risk >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        y_bad,
        predicted_bad,
    )

    precision = precision_score(
        y_bad,
        predicted_bad,
        zero_division=0,
    )

    recall = recall_score(
        y_bad,
        predicted_bad,
        zero_division=0,
    )

    f1 = f1_score(
        y_bad,
        predicted_bad,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_bad,
        bad_risk,
    )

    average_precision = average_precision_score(
        y_bad,
        bad_risk,
    )

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            y_bad,
            bad_risk,
        )
    )

    exact_pr_auc = auc(
        recall_curve,
        precision_curve,
    )

    matrix = confusion_matrix(
        y_bad,
        predicted_bad,
        labels=[1, 0],
    )

    return {
        "accuracy": accuracy,
        "precision_bad": precision,
        "recall_bad": recall,
        "f1_bad": f1,
        "roc_auc": roc_auc,
        "pr_auc": exact_pr_auc,
        "average_precision": average_precision,
        "confusion_matrix": matrix.tolist(),
    }


def build_candidate_models(
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Pipeline]:

    def pipeline_for(model):
        return Pipeline(
            steps=[
                (
                    "preprocessor",
                    make_training_preprocessor(
                        numeric_features,
                        categorical_features,
                    ),
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

    return {
        "Balanced Logistic Regression": pipeline_for(
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            )
        ),

        "Balanced Random Forest": pipeline_for(
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        ),

        "Tuned HistGradientBoosting": pipeline_for(
            HistGradientBoostingClassifier(
                learning_rate=0.03,
                max_iter=300,
                max_leaf_nodes=15,
                min_samples_leaf=50,
                l2_regularization=5.0,
                random_state=42,
            )
        ),
    }

def predict_with_streamlit_model(
    pipeline,
    engineered_features: pd.DataFrame,
    feature_names: list[str],
    threshold: float = 0.50,
):
    """
    Run inference using a fitted Streamlit training pipeline.

    The pipeline must contain the same preprocessing structure
    used during training.
    """

    missing = [
        feature
        for feature in feature_names
        if feature not in engineered_features.columns
    ]

    if missing:
        raise ValueError(
            "Engineered data is missing required model features: "
            + ", ".join(missing)
        )

    X = engineered_features[feature_names].copy()

    probabilities = pipeline.predict_proba(X)

    classes = list(pipeline.classes_)

    if 0 not in classes:
        raise ValueError(
            "The trained model does not contain the expected Bad-risk class 0."
        )

    bad_index = classes.index(0)

    bad_risk = float(probabilities[0, bad_index])

    predicted_class = (
        "Bad"
        if bad_risk >= threshold
        else "Good"
    )

    return predicted_class, bad_risk

# ============================================================
# LOAD PROJECT DATA
# ============================================================

@st.cache_data
def load_project_data():

    performance = pd.read_csv(
        DATA_DIR / "trainperf.csv"
    )

    previous_loans = pd.read_csv(
        DATA_DIR / "trainprevloans.csv"
    )

    demographics = pd.read_csv(
        DATA_DIR / "traindemographics.csv"
    )

    return (
        performance,
        previous_loans,
        demographics,
    )


performance, previous_loans, demographics = load_project_data()

# ============================================================
# PRODUCTION RISK SERVICE
# ============================================================

@st.cache_resource
def load_risk_service() -> RiskPredictionService:
    from src.model_manager import ModelManager


    model_path = (
        ARTIFACT_DIR /
        "superlender_hgb_pipeline.joblib"
    )

    metadata_path = (
        ARTIFACT_DIR /
        "model_metadata.json"
    )

    feature_engineer = FeatureEngineer()
    model_manager = ModelManager(
        model_path=model_path,
        metadata_path=metadata_path
    )

    return RiskPredictionService(
        feature_engineer=feature_engineer,
        model_manager=model_manager
    )

risk_service = load_risk_service()

# ============================================================
# STREAMLIT TRAINING-MODEL SESSION STATE
# ============================================================

if "training_model" not in st.session_state:
    st.session_state["training_model"] = None

if "training_model_name" not in st.session_state:
    st.session_state["training_model_name"] = None

if "training_model_features" not in st.session_state:
    st.session_state["training_model_features"] = None

if "training_model_threshold" not in st.session_state:
    st.session_state["training_model_threshold"] = 0.50

if "training_model_selection_criterion" not in st.session_state:
    st.session_state["training_model_selection_criterion"] = None

if "training_model_run_id" not in st.session_state:
    st.session_state["training_model_run_id"] = 0


def predict_batch_with_streamlit_model(
    pipeline,
    engineered_features: pd.DataFrame,
    feature_names: list[str],
    threshold: float = 0.50,
):
    """Run batch inference using a fitted Streamlit training pipeline."""

    missing = [
        feature
        for feature in feature_names
        if feature not in engineered_features.columns
    ]

    if missing:
        raise ValueError(
            "Engineered data is missing required model features: "
            + ", ".join(missing)
        )

    X = engineered_features[feature_names].copy()
    probabilities = pipeline.predict_proba(X)
    classes = list(pipeline.named_steps["model"].classes_)

    if 0 not in classes:
        raise ValueError(
            "The trained model does not contain the expected Bad-risk class 0."
        )

    bad_index = classes.index(0)
    bad_scores = probabilities[:, bad_index]

    predicted_classes = np.where(
        bad_scores >= threshold,
        "Bad",
        "Good",
    )

    return predicted_classes, bad_scores

def validate_batch_input(
    batch_df: pd.DataFrame,
) -> list[str]:
    """
    Validate current-loan records before batch feature engineering.
    """

    required_columns = {
        "systemloanid",
        "customerid",
        "creationdate",
        "loanamount",
        "totaldue",
        "termdays",
        "loannumber",
        "referredby",
    }

    errors = []

    if batch_df.empty:
        errors.append("The uploaded file contains no application records.")
        return errors

    missing_columns = required_columns - set(batch_df.columns)

    if missing_columns:
        errors.append(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if "good_bad_flag" in batch_df.columns:
        errors.append(
            "The uploaded file contains 'good_bad_flag'. Target labels must "
            "not be supplied to the prediction workflow."
        )

    if "systemloanid" in batch_df.columns:
        if batch_df["systemloanid"].isna().any():
            errors.append("'systemloanid' contains missing values.")
        if batch_df["systemloanid"].duplicated().any():
            errors.append(
                "'systemloanid' must be unique within the uploaded batch."
            )

    if "customerid" in batch_df.columns:
        if batch_df["customerid"].isna().any():
            errors.append("'customerid' contains missing values.")

    for column in ["loanamount", "totaldue", "termdays", "loannumber"]:
        if column in batch_df.columns:
            numeric_values = pd.to_numeric(
                batch_df[column],
                errors="coerce",
            )
            if numeric_values.isna().any():
                errors.append(
                    f"'{column}' contains non-numeric or missing values."
                )

    if "loanamount" in batch_df.columns:
        loanamount = pd.to_numeric(
            batch_df["loanamount"],
            errors="coerce",
        )
        if (loanamount <= 0).fillna(False).any():
            errors.append("'loanamount' must contain only positive values.")

    if "totaldue" in batch_df.columns:
        totaldue = pd.to_numeric(
            batch_df["totaldue"],
            errors="coerce",
        )
        if (totaldue <= 0).fillna(False).any():
            errors.append("'totaldue' must contain only positive values.")

    if {"loanamount", "totaldue"}.issubset(batch_df.columns):
        loanamount = pd.to_numeric(
            batch_df["loanamount"],
            errors="coerce",
        )
        totaldue = pd.to_numeric(
            batch_df["totaldue"],
            errors="coerce",
        )
        if (loanamount > totaldue).fillna(False).any():
            errors.append("'loanamount' must not exceed 'totaldue'.")

    if "loannumber" in batch_df.columns:
        loannumber = pd.to_numeric(
            batch_df["loannumber"],
            errors="coerce",
        )
        if (loannumber <= 0).fillna(False).any():
            errors.append("'loannumber' must contain only positive values.")

    if "creationdate" in batch_df.columns:
        parsed_dates = pd.to_datetime(
            batch_df["creationdate"],
            errors="coerce",
        )
        invalid_dates = parsed_dates.isna().sum()
        if invalid_dates > 0:
            errors.append(
                f"'creationdate' contains {invalid_dates:,} invalid or missing dates."
            )

    return errors

def count_applications_with_observable_history(
    batch_df: pd.DataFrame,
    previous_loans: pd.DataFrame,
) -> int:
    """Count uploaded applications with at least one observable historical loan."""

    current = batch_df[
        ["systemloanid", "customerid", "creationdate"]
    ].copy()

    current["creationdate"] = pd.to_datetime(
        current["creationdate"],
        errors="coerce",
    )

    history = previous_loans[
        ["customerid", "approveddate"]
    ].copy()

    history["approveddate"] = pd.to_datetime(
        history["approveddate"],
        errors="coerce",
    )

    merged = current.merge(
        history,
        on="customerid",
        how="left",
    )

    observable = merged[
        merged["approveddate"] < merged["creationdate"]
    ]

    return int(observable["systemloanid"].nunique())


def load_json_artifact(path: Path) -> dict:
    """
    Load a JSON artifact safely.
    """

    import json

    if not path.exists():
        return {}

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        return (
            data
            if isinstance(data, dict)
            else {}
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {}


def format_metric(
    value,
    percentage: bool = False,
) -> str:
    """
    Format a metric for display.
    """

    if value is None:
        return "—"

    try:
        numeric_value = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return "—"

    if percentage:
        return f"{numeric_value:.1%}"

    return f"{numeric_value:.3f}"


def read_artifact_csv(
    path: Path,
) -> pd.DataFrame:
    """
    Read a CSV artifact safely.
    """

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)

    except (
        OSError,
        pd.errors.ParserError,
    ):
        return pd.DataFrame()

MODEL_METADATA = load_json_artifact(
    ARTIFACT_DIR / "model_metadata.json"
)

MODEL_EVIDENCE = load_json_artifact(
    EVIDENCE_PATH
)

PERMUTATION_IMPORTANCE = read_artifact_csv(
    ARTIFACT_DIR / "permutation_importance.csv"
)

OPERATING_POINTS = read_artifact_csv(
    ARTIFACT_DIR / "final_operating_points.csv"
)

FAIRNESS_AUDIT = read_artifact_csv(FAIRNESS_PATH)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 💳 SuperLender")

    st.caption(
        "Credit Risk Intelligence Platform"
    )

    st.divider()

    page = st.radio(
        "Navigate",
        [
            "Executive Dashboard",
            "Data & EDA",
            "Model Training",
            "Risk Assessment",
            "Batch Intelligence",
            "Model & Evidence",
        ],
    )

    st.divider()

    st.caption("Production model")

    st.write(
        "HistGradientBoostingClassifier"
    )

    st.caption("Operating threshold")

    st.write("23%")

    st.divider()

    st.caption(
        "Model output is a risk estimate, "
        "not a guaranteed lending decision."
    )

# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if page == "Executive Dashboard":

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    st.markdown(
        '<div class="brand-title">SuperLender</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="brand-subtitle">'
        'Credit Risk Intelligence Platform'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "A credit-risk system that combines current borrowing "
        "characteristics with application-time customer history "
        "to estimate default risk."
    )


    # --------------------------------------------------------
    # Portfolio metrics
    # --------------------------------------------------------

    total_applications = len(performance)

    total_bad = (
        performance["good_bad_flag"] == "Bad"
    ).sum()

    total_good = (
        performance["good_bad_flag"] == "Good"
    ).sum()

    bad_rate = (
        total_bad / total_applications
    )

    total_history = len(previous_loans)


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        metric_card(
            "Current applications",
            f"{total_applications:,}",
            "Training applications",
        )


    with col2:

        metric_card(
            "Observed Bad outcomes",
            f"{total_bad:,}",
            f"{bad_rate:.1%} of applications",
        )


    with col3:

        metric_card(
            "Observed Good outcomes",
            f"{total_good:,}",
            f"{1 - bad_rate:.1%} of applications",
        )


    with col4:

        metric_card(
            "Historical loan records",
            f"{total_history:,}",
            "Previous-loan observations",
        )


    # ========================================================
    # OBSERVED RISK PATTERN
    # ========================================================

    section_header(
        "Observed risk pattern",
        "Historical repayment behaviour shows a measurable "
        "relationship with observed Bad outcomes in the "
        "training data.",
    )


    late_rate_analysis = create_late_rate_analysis(
        previous_loans,
        performance,
    )


    # --------------------------------------------------------
    # Create late-rate bands
    # --------------------------------------------------------

    late_rate_analysis["late_rate_band"] = pd.cut(
        late_rate_analysis["historical_late_rate"],
        bins=[
            -0.001,
            0,
            0.25,
            0.50,
            0.75,
            1.001,
        ],
        labels=[
            "0%",
            "0–25%",
            "25–50%",
            "50–75%",
            "75–100%",
        ],
    )


    # --------------------------------------------------------
    # Summarise observed Bad rate by band
    # --------------------------------------------------------

    risk_summary = (
        late_rate_analysis
        .dropna(
            subset=["historical_late_rate"]
        )
        .groupby(
            "late_rate_band",
            observed=False,
        )
        .agg(
            applications=(
                "good_bad_flag",
                "size",
            ),
            bad_rate=(
                "good_bad_flag",
                lambda x: (x == "Bad").mean(),
            ),
        )
        .reset_index()
    )


    left, right = st.columns(
        [2, 1]
    )


    # --------------------------------------------------------
    # Chart
    # --------------------------------------------------------

    with left:

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        ax.bar(
            risk_summary[
                "late_rate_band"
            ].astype(str),
            risk_summary[
                "bad_rate"
            ],
        )

        ax.set_xlabel(
            "Historical late-repayment rate"
        )

        ax.set_ylabel(
            "Observed Bad rate"
        )

        ax.set_title(
            "Observed Bad rate by historical "
            "late-repayment behaviour"
        )

        ax.set_ylim(
            0,
            risk_summary["bad_rate"].max()
            * 1.25,
        )

        for i, value in enumerate(
            risk_summary["bad_rate"]
        ):

            ax.text(
                i,
                value + 0.005,
                f"{value:.1%}",
                ha="center",
                fontsize=9,
            )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    with right:

        st.markdown(
            "### What this tells us"
        )

        st.write(
            "Customers associated with higher proportions "
            "of late historical repayments show higher "
            "observed Bad rates across these training-data "
            "groups."
        )

        st.caption(
            "Association in the training data does not "
            "establish causation."
        )


    # ========================================================
    # HOW SUPERLENDER WORKS
    # ========================================================

    section_header(
        "How SuperLender works",
        "From raw lending records to application-time "
        "risk intelligence.",
    )


    workflow = st.columns(9)


    with workflow[0]:

        st.markdown("### 01")

        st.write(
            "**Current application**"
        )


    with workflow[1]:

        st.markdown("### →")


    with workflow[2]:

        st.markdown("### 02")

        st.write(
            "**Customer history**"
        )


    with workflow[3]:

        st.markdown("### →")


    with workflow[4]:

        st.markdown("### 03")

        st.write(
            "**Risk features**"
        )


    with workflow[5]:

        st.markdown("### →")


    with workflow[6]:

        st.markdown("### 04")

        st.write(
            "**Risk model**"
        )


    with workflow[7]:

        st.markdown("### →")


    with workflow[8]:

        st.markdown("### 05")

        st.write(
            "**Risk estimate**"
        )


    # ========================================================
    # WHAT THE MODEL USES
    # ========================================================

    section_header(
        "What the model uses to estimate risk",
        "SuperLender combines current borrowing characteristics "
        "with historical behavioural evidence available at "
        "application time.",
    )


    left, right = st.columns(2)


    with left:

        st.markdown(
            "#### Current borrowing profile"
        )

        st.write(
            "Current loan characteristics describe the "
            "size, cost and structure of the borrowing request."
        )


    with right:

        st.markdown(
            "#### Behavioural evidence"
        )

        st.write(
            "Historical repayment behaviour provides evidence "
            "about how the customer has interacted with "
            "previous loans."
        )


    # ========================================================
    # DATA INTEGRITY PRINCIPLES
    # ========================================================

    section_header(
        "Data integrity principles",
        "The production pipeline is designed around the "
        "prediction boundary.",
    )


    a, b, c = st.columns(3)


    with a:

        st.markdown(
            "### Application-time information"
        )

        st.write(
            "Historical repayment information is only used "
            "when it was observable before the current "
            "loan application."
        )


    with b:

        st.markdown(
            "### One row per prediction"
        )

        st.write(
            "Historical loans are aggregated to the "
            "customer/current-loan level before modeling."
        )


    with c:

        st.markdown(
            "### Reproducible predictions"
        )

        st.write(
            "The deployed feature pipeline reproduces "
            "the training feature contract before inference."
        )


    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown(
        """
        <div class="footer">
            SuperLender • Credit Risk Data Science Project
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DATA & EDA
# ============================================================

elif page == "Data & EDA":

    st.markdown(
        '<div class="brand-title">Data & EDA</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="brand-subtitle">'
        'Understand the lending data before modelling'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Explore the three source datasets, assess their quality, "
        "understand how they relate, and inspect the current-loan-level "
        "working dataset used as the foundation for risk analysis."
    )

    # ========================================================
    # DATASET DEFINITIONS
    # ========================================================

    datasets = {
        "Current applications": {
            "data": performance,
            "description": (
                "Current loan applications and their observed "
                "Good/Bad outcomes."
            ),
            "keys": ["systemloanid", "customerid"],
            "dates": ["creationdate", "approveddate"],
        },
        "Historical loans": {
            "data": previous_loans,
            "description": (
                "Previous loans associated with customers represented "
                "in the current applications."
            ),
            "keys": ["systemloanid", "customerid"],
            "dates": [
                "creationdate",
                "approveddate",
                "closeddate",
                "firstduedate",
                "firstrepaiddate",
            ],
        },
        "Demographics": {
            "data": demographics,
            "description": (
                "Customer demographic and banking information."
            ),
            "keys": ["customerid"],
            "dates": ["birthdate"],
        },
    }

    # ========================================================
    # OVERVIEW
    # ========================================================

    section_header(
        "Data overview",
        "The project combines current applications, customer "
        "demographics and historical loan behaviour.",
    )

    overview_1, overview_2, overview_3 = st.columns(3)

    with overview_1:
        metric_card(
            "Current applications",
            f"{len(performance):,}",
            "One row per current loan",
        )

    with overview_2:
        metric_card(
            "Historical loans",
            f"{len(previous_loans):,}",
            "One row per previous loan",
        )

    with overview_3:
        metric_card(
            "Demographic records",
            f"{len(demographics):,}",
            "Customer-level records before deduplication",
        )

    # ========================================================
    # DATASET INSPECTION
    # ========================================================

    section_header(
        "Dataset inspection",
        "Inspect samples, structure, data types and missing values "
        "for each source independently.",
    )

    selected_dataset = st.selectbox(
        "Dataset",
        list(datasets.keys()),
        key="eda_dataset_selector",
    )

    selected_config = datasets[selected_dataset]
    selected_df = selected_config["data"]

    st.caption(selected_config["description"])

    metric_a, metric_b, metric_c = st.columns(3)

    with metric_a:
        metric_card(
            "Rows",
            f"{len(selected_df):,}",
        )

    with metric_b:
        metric_card(
            "Columns",
            f"{len(selected_df.columns):,}",
        )

    with metric_c:
        metric_card(
            "Exact duplicate rows",
            f"{selected_df.duplicated().sum():,}",
        )

    inspect_tab, quality_tab, dates_tab = st.tabs(
        [
            "Sample & structure",
            "Missingness & keys",
            "Date ranges",
        ]
    )

    # --------------------------------------------------------
    # SAMPLE & STRUCTURE
    # --------------------------------------------------------

    with inspect_tab:

        st.markdown("#### Sample")

        st.dataframe(
            selected_df.head(10),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Data types")

        dtype_table = (
            selected_df.dtypes
            .astype(str)
            .rename("dtype")
            .to_frame()
        )

        dtype_table["non_null"] = (
            selected_df.notna().sum()
        )

        dtype_table["missing"] = (
            selected_df.isna().sum()
        )

        st.dataframe(
            dtype_table,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    with quality_tab:

        st.markdown("#### Missing values")

        display_missingness(selected_df)

        st.markdown("#### Key quality")

        display_key_quality(
            selected_df,
            selected_config["keys"],
        )

    # --------------------------------------------------------
    # DATE RANGES
    # --------------------------------------------------------

    with dates_tab:

        st.markdown("#### Available date ranges")

        display_date_ranges(
            selected_df,
            selected_config["dates"],
        )

    # ========================================================
    # DATASET RELATIONSHIPS
    # ========================================================

    section_header(
        "Dataset relationships",
        "The current application is the prediction grain. "
        "Historical loans are one-to-many at customer level, while "
        "demographics are intended to contribute customer-level information.",
    )

    rel1, rel2, rel3 = st.columns(3)

    with rel1:

        st.markdown("### Current application")

        st.write(
            "**Grain:** one row per current loan"
        )

        st.write(
            "`systemloanid` identifies the loan and "
            "`customerid` connects the application to customer information."
        )

    with rel2:

        st.markdown("### Customer information")

        st.write(
            "**Grain:** customer-level"
        )

        st.write(
            "`customerid` connects demographic information "
            "to the current application."
        )

    with rel3:

        st.markdown("### Historical behaviour")

        st.write(
            "**Grain:** one row per previous loan"
        )

        st.write(
            "A customer may have multiple historical loans, "
            "so historical records must be aggregated before "
            "they can be attached to one current application."
        )

    st.markdown("---")

    st.markdown(
        """
        **Relational flow**

        `Current application`
        → `customerid`
        → `Demographics`

        `Current application`
        → `customerid`
        → `Historical loans`
        → `Customer-level historical features`
        """
    )

    # ========================================================
    # MERGED WORKING DATASET
    # ========================================================

    section_header(
        "Current-loan working dataset",
        "A relational preview showing how customer-level demographic "
        "information and aggregated historical-loan information can "
        "be attached without changing the current-loan grain.",
    )

    working_dataset, merge_audit = (
        build_working_dataset_preview(
            performance=performance,
            previous_loans=previous_loans,
            demographics=demographics,
        )
    )

    audit_1, audit_2, audit_3, audit_4 = st.columns(4)

    with audit_1:
        metric_card(
            "Current rows",
            f"{merge_audit['current_rows_before_merge']:,}",
            "Before merging",
        )

    with audit_2:
        metric_card(
            "Demographic matches",
            f"{merge_audit['demographic_matched']:,}",
            "Current applications",
        )

    with audit_3:
        metric_card(
            "No demographics",
            f"{merge_audit['demographic_unmatched']:,}",
            "Handled as missing information",
        )

    with audit_4:
        metric_card(
            "Current rows after merge",
            f"{merge_audit['rows_after_history_aggregation']:,}",
            "Expected one row per loan",
        )

    st.markdown("#### Merge integrity")

    merge_check_1, merge_check_2, merge_check_3 = st.columns(3)

    with merge_check_1:

        if (
            merge_audit["rows_after_demographics_merge"]
            == merge_audit["current_rows_before_merge"]
        ):
            st.success(
                "Demographics merge preserved row count."
            )
        else:
            st.error(
                "Demographics merge changed the current-loan row count."
            )

    with merge_check_2:

        if merge_audit["final_duplicate_systemloanid"] == 0:
            st.success(
                "No duplicate current loan IDs after aggregation."
            )
        else:
            st.error(
                "Duplicate current loan IDs detected."
            )

    with merge_check_3:

        st.info(
            f"{merge_audit['current_with_history']:,} current "
            "applications have historical loan records; "
            f"{merge_audit['current_without_history']:,} do not."
        )

    st.markdown("#### Working dataset preview")

    preview_columns = [
        "systemloanid",
        "customerid",
        "creationdate",
        "good_bad_flag",
        "historical_loan_count",
    ]

    preview_columns = [
        column
        for column in preview_columns
        if column in working_dataset.columns
    ]

    st.dataframe(
        working_dataset[preview_columns].head(20),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "This preview demonstrates the relational structure only. "
        "The production feature pipeline applies additional "
        "application-time temporal rules before historical "
        "repayment features are used for inference."
    )

    # ========================================================
    # EDA DATASET
    # ========================================================

    @st.cache_data
    def load_eda_dataset(
        performance_df: pd.DataFrame,
        previous_loans_df: pd.DataFrame,
        demographics_df: pd.DataFrame,
    ) -> pd.DataFrame:

        return build_eda_dataset(
            performance=performance_df,
            previous_loans=previous_loans_df,
            demographics=demographics_df,
        )


    eda_data = load_eda_dataset(
        performance,
        previous_loans,
        demographics,
    )

    # ========================================================
    # CANONICAL PRODUCTION FEATURE CATALOG
    # ========================================================

    production_numeric_features = (
        FeatureEngineer.NUMERIC_FEATURES.copy()
    )

    production_categorical_features = (
        FeatureEngineer.CATEGORICAL_FEATURES.copy()
    )

    production_feature_columns = (
        FeatureEngineer.MODEL_FEATURES.copy()
    )

    available_production_features = [
        column
        for column in production_feature_columns
        if column in eda_data.columns
    ]

    derived_feature_catalog = pd.DataFrame([
        {
            "Feature": "loan_cost",
            "Type": "Derived numerical",
            "Definition": "totaldue − loanamount",
            "Risk meaning": "Amount charged above principal.",
            "Available": "At application",
        },
        {
            "Feature": "loan_cost_rate",
            "Type": "Derived numerical",
            "Definition": "loan_cost / loanamount",
            "Risk meaning": "Relative cost of the current loan.",
            "Available": "At application",
        },
        {
            "Feature": "creation_hour",
            "Type": "Derived temporal",
            "Definition": "Hour extracted from creationdate",
            "Risk meaning": "Application timing signal.",
            "Available": "At application",
        },
        {
            "Feature": "creation_day_of_week",
            "Type": "Derived temporal",
            "Definition": "Day extracted from creationdate",
            "Risk meaning": "Application timing signal.",
            "Available": "At application",
        },
        {
            "Feature": "age_at_application",
            "Type": "Derived demographic",
            "Definition": "Age from birthdate and creationdate",
            "Risk meaning": "Customer age at application.",
            "Available": "At application when birthdate exists",
        },
        {
            "Feature": "referredby_missing",
            "Type": "Derived indicator",
            "Definition": "1 when referral is missing",
            "Risk meaning": "Represents absence of referral information.",
            "Available": "At application",
        },
        {
            "Feature": "historical_loan_count",
            "Type": "Historical aggregate",
            "Definition": "Count of prior loans",
            "Risk meaning": "Borrowing depth before current loan.",
            "Available": "Application-time history",
        },
        {
            "Feature": "observed_late_repayment_rate",
            "Type": "Historical aggregate",
            "Definition": "Late observed repayments / observed repayments",
            "Risk meaning": "Historical repayment behaviour.",
            "Available": "Only repayments observable before application",
        },
        {
            "Feature": "days_since_last_loan",
            "Type": "Historical temporal",
            "Definition": "Current application date − latest historical approval",
            "Risk meaning": "Recency of previous borrowing.",
            "Available": "Application-time history",
        },
    ])

    # ========================================================
    # WRANGLING & FEATURE WORKSPACE
    # ========================================================

    section_header(
        "Wrangling & feature workspace",
        "Configure an analytical feature set and inspect how "
        "wrangling choices affect the working dataset.",
    )

    st.info(
        "The analytical workspace is built from the source datasets and "
        "selected application-time derived features. It is intentionally "
        "separate from the frozen production model pipeline used for inference."
    )

    # ========================================================
    # COLUMN SELECTION
    # ========================================================

    st.markdown("### Feature selection")

    selected_features = st.multiselect(
        "Features available for the analysis workspace",
        options=available_production_features,
        default=available_production_features,
        key="eda_selected_features",
    )

    if not selected_features:

        st.warning(
            "Select at least one feature for the analysis workspace."
        )

    else:

        st.caption(
            f"{len(selected_features)} of "
            f"{len(available_production_features)} available production "
            "features are currently selected."
        )

    # ========================================================
    # COLUMN DROPPING
    # ========================================================

    st.markdown("### Columns to drop")

    columns_to_drop = st.multiselect(
        "Temporarily exclude columns from the analytical dataset",
        options=selected_features,
        default=[],
        key="eda_columns_to_drop",
    )

    active_features = [
        feature
        for feature in selected_features
        if feature not in columns_to_drop
    ]

    st.caption(
        f"{len(active_features)} active features remain after "
        f"dropping {len(columns_to_drop)} selected columns."
    )

    # ========================================================
    # MISSING-VALUE TREATMENT
    # ========================================================

    st.markdown("### Missing-value treatment")

    missing_left, missing_right = st.columns(2)

    with missing_left:

        numeric_missing_strategy = st.selectbox(
            "Numerical features",
            [
                "Keep missing values",
                "Median imputation",
                "Drop rows with missing values",
            ],
            index=1,
            key="eda_numeric_missing_strategy",
        )

    with missing_right:

        categorical_missing_strategy = st.selectbox(
            "Categorical features",
            [
                "Keep missing values",
                "Fill with Unknown",
                "Most frequent category",
                "Drop rows with missing values",
            ],
            index=1,
            key="eda_categorical_missing_strategy",
        )

    st.caption(
        "These settings affect the analytical preview only. "
        "Production preprocessing remains train-fitted and "
        "is reused by inference."
    )

    # ========================================================
    # DERIVED FEATURE INSPECTION
    # ========================================================

    st.markdown("### Derived features")

    st.write(
        "These features are constructed from information that was "
        "available at the application boundary, subject to the "
        "historical temporal rules used by the project."
    )

    st.dataframe(
        derived_feature_catalog,
        use_container_width=True,
        hide_index=True,
    )

    selected_derived_feature = st.selectbox(
        "Inspect a derived feature",
        derived_feature_catalog["Feature"].tolist(),
        key="eda_derived_feature",
    )

    derived_row = derived_feature_catalog[
        derived_feature_catalog["Feature"]
        == selected_derived_feature
    ].iloc[0]

    d1, d2, d3 = st.columns(3)

    with d1:
        st.markdown("**Definition**")
        st.write(derived_row["Definition"])

    with d2:
        st.markdown("**Risk meaning**")
        st.write(derived_row["Risk meaning"])

    with d3:
        st.markdown("**Availability**")
        st.write(derived_row["Available"])

    if selected_derived_feature in eda_data.columns:

        st.markdown(
            f"#### `{selected_derived_feature}` preview"
        )

        derived_preview = eda_data[
            [
                "systemloanid",
                "customerid",
                selected_derived_feature,
            ]
        ].head(10)

        st.dataframe(
            derived_preview,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # APPLY WRANGLING
    # ========================================================

    st.markdown("### Apply analytical configuration")

    def apply_wrangling_preview(
        source_df: pd.DataFrame,
        features: list[str],
        numeric_strategy: str,
        categorical_strategy: str,
    ) -> pd.DataFrame:

        result = source_df[
            [
                column
                for column in features
                if column in source_df.columns
            ]
        ].copy()

        numeric_columns = [
            column
            for column in production_numeric_features
            if column in result.columns
        ]

        categorical_columns = [
            column
            for column in production_categorical_features
            if column in result.columns
        ]

        # ----------------------------------------------
        # Numerical treatment
        # ----------------------------------------------

        if numeric_strategy == "Median imputation":

            for column in numeric_columns:

                median = result[column].median()

                if pd.notna(median):
                    result[column] = result[column].fillna(
                        median
                    )

        elif numeric_strategy == "Drop rows with missing values":

            result = result.dropna(
                subset=numeric_columns
            )

        # ----------------------------------------------
        # Categorical treatment
        # ----------------------------------------------

        if categorical_strategy == "Fill with Unknown":

            result[categorical_columns] = (
                result[categorical_columns]
                .fillna("Unknown")
            )

        elif categorical_strategy == "Most frequent category":

            for column in categorical_columns:

                mode = result[column].mode(
                    dropna=True
                )

                if not mode.empty:

                    result[column] = (
                        result[column]
                        .fillna(mode.iloc[0])
                    )

        elif categorical_strategy == "Drop rows with missing values":

            result = result.dropna(
                subset=categorical_columns
            )

        return result

    configured_features = apply_wrangling_preview(
        source_df=eda_data,
        features=active_features,
        numeric_strategy=numeric_missing_strategy,
        categorical_strategy=categorical_missing_strategy,
    )

    # ========================================================
    # RESULT SUMMARY
    # ========================================================

    st.markdown("### Resulting analytical dataset")

    r1, r2, r3 = st.columns(3)

    with r1:

        metric_card(
            "Rows",
            f"{len(configured_features):,}",
            f"From {len(eda_data):,} source rows",
        )

    with r2:

        metric_card(
            "Features",
            f"{len(configured_features.columns):,}",
            "After current selections",
        )

    with r3:

        remaining_missing = int(
            configured_features.isna().sum().sum()
        )

        metric_card(
            "Remaining missing cells",
            f"{remaining_missing:,}",
            "After selected treatment",
        )

    st.dataframe(
        configured_features.head(20),
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # FEATURE TYPE SUMMARY
    # ========================================================

    numerical_active = [
        column
        for column in production_numeric_features
        if column in configured_features.columns
    ]

    categorical_active = [
        column
        for column in production_categorical_features
        if column in configured_features.columns
    ]

    f1, f2 = st.columns(2)

    with f1:

        st.markdown("#### Numerical features")

        st.write(
            f"{len(numerical_active):,} numerical features"
        )

        st.code(
            "\n".join(numerical_active)
            if numerical_active
            else "None"
        )

    with f2:

        st.markdown("#### Categorical features")

        st.write(
            f"{len(categorical_active):,} categorical features"
        )

        st.code(
            "\n".join(categorical_active)
            if categorical_active
            else "None"
        )

    # ========================================================
    # MODELING DATASET NOTE
    # ========================================================

    st.caption(
        "This is an analytical preview, not a replacement for the frozen "
        "production feature pipeline. Production inference continues to use "
        "the saved 34-feature training contract and train-fitted preprocessing."
    )

    # Save current configuration for later Streamlit pages.
    st.session_state["eda_active_features"] = active_features
    st.session_state["eda_wrangled_data"] = configured_features

    # ========================================================
    # EXPLORATORY DATA ANALYSIS
    # ========================================================

    section_header(
        "Risk exploration",
        "Interactively examine how current-loan, historical-behaviour, "
        "demographic and application-timing variables relate to observed "
        "Good/Bad outcomes.",
    )

    # ========================================================
    # TARGET DISTRIBUTION
    # ========================================================

    st.markdown("### Target distribution")

    total_applications = len(eda_data)

    good_count = (
        eda_data["good_bad_flag"] == "Good"
    ).sum()

    bad_count = (
        eda_data["good_bad_flag"] == "Bad"
    ).sum()

    good_rate = good_count / total_applications
    bad_rate = bad_count / total_applications

    t1, t2, t3 = st.columns(3)

    with t1:
        metric_card(
            "Good applications",
            f"{good_count:,}",
            f"{good_rate:.1%} of current applications",
        )

    with t2:
        metric_card(
            "Bad applications",
            f"{bad_count:,}",
            f"{bad_rate:.1%} of current applications",
        )

    with t3:
        metric_card(
            "Bad-class minority",
            f"{bad_rate:.1%}",
            "Observed target prevalence",
        )

    target_counts = pd.DataFrame(
        {
            "Outcome": ["Good", "Bad"],
            "Applications": [
                good_count,
                bad_count,
            ],
        }
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.bar(
        target_counts["Outcome"],
        target_counts["Applications"],
    )

    ax.set_ylabel("Applications")
    ax.set_title("Current-loan outcome distribution")

    for bar, value in zip(
        bars,
        target_counts["Applications"],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 40,
            f"{value:,}",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()

    st.pyplot(fig)

    figure_download_button(
        fig,
        "superlender_target_distribution.png",
    )

    plt.close(fig)

    st.info(
        f"Observed Bad outcomes represent {bad_rate:.1%} of the "
        f"{total_applications:,} current applications. This class "
        "imbalance means accuracy alone does not describe how well "
        "the model identifies Bad loans."
    )

    # ========================================================
    # INTERACTIVE VARIABLE ANALYSIS
    # ========================================================

    st.markdown("### Variable versus observed Bad rate")

    continuous_variables = {
        "Loan amount": "loanamount",
        "Loan cost": "loan_cost",
        "Loan-cost rate": "loan_cost_rate",
        "Age at application": "age_at_application",
        "Historical loan count": "historical_loan_count",
        "Days since last historical loan": "days_since_last_loan",
        "Observed late-repayment rate": (
            "observed_late_repayment_rate"
        ),
        "Average observed repayment delay": (
            "avg_observed_repayment_delay_days"
        ),
    }

    categorical_variables = {
        "Loan term": "termdays",
        "Loan number": "loannumber",
        "Application day": "creation_day_of_week",
        "Application hour": "creation_hour",
        "Bank account type": "bank_account_type",
        "Employment status": "employment_status_clients",
        "Bank": "bank_name_clients",
        "Education": "level_of_education_clients",
        "Referral missingness": "referredby_missing",
        "Historical depth": "history_depth_band",
        "Late-repayment band": "late_rate_band",
    }

    continuous_variables = {
        label: column
        for label, column in continuous_variables.items()
        if column in eda_data.columns
    }

    categorical_variables = {
        label: column
        for label, column in categorical_variables.items()
        if column in eda_data.columns
    }

    analysis_type = st.radio(
        "Analysis type",
        [
            "Continuous variable",
            "Categorical variable",
        ],
        horizontal=True,
        key="eda_analysis_type",
    )

    if analysis_type == "Continuous variable":

        selected_label = st.selectbox(
            "Variable",
            list(continuous_variables.keys()),
            key="eda_continuous_variable",
        )

        selected_column = continuous_variables[
            selected_label
        ]

        analysis_df = eda_data[
            [
                selected_column,
                "good_bad_flag",
            ]
        ].dropna()

        if analysis_df.empty:

            st.warning(
                "There are no usable observations for this variable."
            )

        else:

            # -----------------------------------------------
            # Summary statistics
            # -----------------------------------------------

            st.markdown("#### Summary statistics")

            descriptive = (
                analysis_df[selected_column]
                .describe()
                .rename("value")
                .to_frame()
            )

            st.dataframe(
                descriptive,
                use_container_width=True,
            )

            # -----------------------------------------------
            # Variable-aware risk bands
            # -----------------------------------------------

            if selected_column == "loanamount":

                risk_bins = [
                    -1,
                    10000,
                    20000,
                    30000,
                    40000,
                    50000,
                    float("inf"),
                ]

                risk_labels = [
                    "₦10k–<₦20k",
                    "₦20k–<₦30k",
                    "₦30k–<₦40k",
                    "₦40k–<₦50k",
                    "₦50k–<₦60k",
                    "₦60k+",
                ]

                analysis_df["risk_band"] = pd.cut(
                    analysis_df[selected_column],
                    bins=risk_bins,
                    labels=risk_labels,
                    right=False,
                )

            elif selected_column == "loannumber":

                analysis_df["risk_band"] = (
                    analysis_df[selected_column]
                    .astype(int)
                    .astype(str)
                )

            elif selected_column == "creation_hour":

                analysis_df["risk_band"] = (
                    analysis_df[selected_column]
                    .astype(int)
                    .astype(str)
                    + ":00"
                )

            elif selected_column == "loan_cost_rate":

                analysis_df["risk_band"] = pd.cut(
                    analysis_df[selected_column],
                    bins=[
                        -float("inf"),
                        0.15,
                        0.20,
                        0.25,
                        0.30,
                        float("inf"),
                    ],
                    labels=[
                        "<15%",
                        "15–20%",
                        "20–25%",
                        "25–30%",
                        "30%+",
                    ],
                )

            elif selected_column == "observed_late_repayment_rate":

                analysis_df["risk_band"] = pd.cut(
                    analysis_df[selected_column],
                    bins=[
                        -0.001,
                        0,
                        0.25,
                        0.50,
                        0.75,
                        1.001,
                    ],
                    labels=[
                        "0%",
                        "0–25%",
                        "25–50%",
                        "50–75%",
                        "75–100%",
                    ],
                )

            else:

                try:

                    analysis_df["risk_band"] = pd.qcut(
                        analysis_df[selected_column],
                        q=5,
                        duplicates="drop",
                    )

                except ValueError:

                    analysis_df["risk_band"] = (
                        analysis_df[selected_column].astype(str)
                    )

            summary = create_bad_rate_summary(
                analysis_df,
                "risk_band",
                minimum_group_size=20,
            )

            if not summary.empty:

                fig, ax = plt.subplots(
                    figsize=(9, 4.5)
                )

                bars = ax.bar(
                    summary["risk_band"].astype(str),
                    summary["bad_rate"],
                )

                ax.set_xlabel(
                    selected_label
                )

                ax.set_ylabel(
                    "Observed Bad rate"
                )

                ax.set_title(
                    f"Observed Bad rate across "
                    f"{selected_label.lower()} ranges"
                )

                add_percentage_labels(
                    ax,
                    bars,
                    summary["bad_rate"],
                )

                ax.set_ylim(
                    0,
                    max(
                        summary["bad_rate"]
                    ) * 1.25,
                )

                plt.xticks(rotation=20)

                plt.tight_layout()

                st.pyplot(fig)

                figure_download_button(
                    fig,
                    "superlender_variable_bad_rate.png",
                )

                plt.close(fig)

                observation_and_inference(
                    summary,
                    "risk_band",
                    selected_label,
                )

    else:

        selected_label = st.selectbox(
            "Variable",
            list(categorical_variables.keys()),
            key="eda_categorical_variable",
        )

        selected_column = categorical_variables[
            selected_label
        ]

        analysis_df = eda_data[
            [
                selected_column,
                "good_bad_flag",
            ]
        ].dropna()

        if analysis_df.empty:

            st.warning(
                "There are no usable observations for this variable."
            )

        else:

            summary = create_bad_rate_summary(
                analysis_df,
                selected_column,
                minimum_group_size=10,
            )

            if summary.empty:

                st.info(
                    "There are not enough observations for a "
                    "stable comparison."
                )

            else:

                # Limit very high-cardinality categories
                # to the most frequent groups.
                if len(summary) > 20:

                    top_groups = (
                        summary
                        .sort_values(
                            "applications",
                            ascending=False,
                        )
                        .head(20)[selected_column]
                    )

                    summary = summary[
                        summary[selected_column].isin(
                            top_groups
                        )
                    ].copy()

                fig, ax = plt.subplots(
                    figsize=(9, 4.5)
                )

                plot_summary = summary.sort_values(
                    "bad_rate",
                    ascending=True,
                )

                bars = ax.barh(
                    plot_summary[selected_column].astype(str),
                    plot_summary["bad_rate"],
                )

                ax.set_xlabel(
                    "Observed Bad rate"
                )

                ax.set_ylabel(
                    selected_label
                )

                ax.set_title(
                    f"Observed Bad rate by "
                    f"{selected_label.lower()}"
                )

                for bar, value in zip(
                    bars,
                    plot_summary["bad_rate"],
                ):

                    ax.text(
                        value + 0.005,
                        bar.get_y()
                        + bar.get_height() / 2,
                        f"{value:.1%}",
                        va="center",
                        fontsize=8,
                    )

                ax.set_xlim(
                    0,
                    max(
                        plot_summary["bad_rate"]
                    ) * 1.25,
                )

                plt.tight_layout()

                st.pyplot(fig)

                figure_download_button(
                    fig,
                    "superlender_categorical_bad_rate.png",
                )

                plt.close(fig)

                st.dataframe(
                    summary.sort_values(
                        "bad_rate",
                        ascending=False,
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                observation_and_inference(
                    summary,
                    selected_column,
                    selected_label,
                )

    # ========================================================
    # HISTORICAL BEHAVIOUR
    # ========================================================

    section_header(
        "Historical repayment behaviour",
        "Historical repayment information is restricted to events "
        "that were observable before the current application.",
    )

    historical_analysis = (
        eda_data[
            [
                "late_rate_band",
                "good_bad_flag",
            ]
        ]
        .dropna()
    )

    historical_summary = create_bad_rate_summary(
        historical_analysis,
        "late_rate_band",
        minimum_group_size=20,
    )

    if not historical_summary.empty:

        fig, ax = plt.subplots(
            figsize=(9, 4.5)
        )

        bars = ax.bar(
            historical_summary[
                "late_rate_band"
            ].astype(str),
            historical_summary[
                "bad_rate"
            ],
        )

        ax.set_xlabel(
            "Historical late-repayment rate"
        )

        ax.set_ylabel(
            "Observed Bad rate"
        )

        ax.set_title(
            "Observed Bad rate by historical repayment behaviour"
        )

        add_percentage_labels(
            ax,
            bars,
            historical_summary["bad_rate"],
        )

        ax.set_ylim(
            0,
            max(
                historical_summary["bad_rate"]
            ) * 1.25,
        )

        plt.tight_layout()

        st.pyplot(fig)

        figure_download_button(
            fig,
            "superlender_historical_late_rate.png",
        )

        plt.close(fig)

        observation_and_inference(
            historical_summary,
            "late_rate_band",
            "historical late-repayment rate",
        )

    # ========================================================
    # MULTIVARIATE ANALYSIS
    # ========================================================

    section_header(
        "Multivariate risk analysis",
        "Examine historical repayment behaviour together with "
        "historical borrowing depth.",
    )

    multivariate_df = eda_data[
        [
            "history_depth_band",
            "late_rate_band",
            "good_bad_flag",
        ]
    ].dropna()

    if not multivariate_df.empty:

        heatmap_data = pd.pivot_table(
            multivariate_df,
            index="history_depth_band",
            columns="late_rate_band",
            values="good_bad_flag",
            aggfunc=lambda x: (
                x == "Bad"
            ).mean(),
            observed=False,
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        image = ax.imshow(
            heatmap_data.values,
            aspect="auto",
        )

        ax.set_xticks(
            range(len(heatmap_data.columns))
        )

        ax.set_xticklabels(
            heatmap_data.columns
        )

        ax.set_yticks(
            range(len(heatmap_data.index))
        )

        ax.set_yticklabels(
            heatmap_data.index
        )

        ax.set_xlabel(
            "Historical late-repayment rate"
        )

        ax.set_ylabel(
            "Historical loan count"
        )

        ax.set_title(
            "Observed Bad rate across historical depth "
            "and repayment behaviour"
        )

        for row_index in range(
            heatmap_data.shape[0]
        ):

            for column_index in range(
                heatmap_data.shape[1]
            ):

                value = heatmap_data.iloc[
                    row_index,
                    column_index,
                ]

                if pd.notna(value):

                    ax.text(
                        column_index,
                        row_index,
                        f"{value:.1%}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

        fig.colorbar(
            image,
            ax=ax,
            label="Observed Bad rate",
        )

        plt.tight_layout()

        st.pyplot(fig)

        figure_download_button(
            fig,
            "superlender_multivariate_risk_analysis.png",
        )

        plt.close(fig)

        st.markdown("#### Observation")

        st.write(
            "Observed Bad rates vary across combinations of "
            "historical borrowing depth and late-repayment behaviour. "
            "Higher late-repayment proportions tend to coincide with "
            "higher observed Bad rates across several history-depth "
            "groups."
        )

        st.markdown("#### Inference")

        st.write(
            "The combined pattern suggests that repayment behaviour "
            "contains risk information beyond simply counting previous "
            "loans. The relationship remains associative rather than "
            "causal, and some cells contain fewer observations than others."
        )


    # ========================================================
    # DATA QUALITY & DIAGNOSTICS
    # ========================================================

    section_header(
        "Data quality & diagnostics",
        "Investigate missingness, key integrity, numerical outliers "
        "and logical consistency before relying on the data for modelling.",
    )

    quality_dataset = st.selectbox(
        "Dataset for quality diagnostics",
        [
            "Current applications",
            "Historical loans",
            "Demographics",
        ],
        key="quality_dataset_selector",
    )

    quality_map = {
        "Current applications": performance,
        "Historical loans": previous_loans,
        "Demographics": demographics,
    }

    quality_df = quality_map[
        quality_dataset
    ]

    # ========================================================
    # MISSINGNESS
    # ========================================================

    st.markdown("### Missingness")

    missing_counts = (
        quality_df
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing_only = missing_counts[
        missing_counts > 0
    ]

    if missing_only.empty:

        st.success(
            "No missing values were detected in this dataset."
        )

    else:

        missing_summary = pd.DataFrame(
            {
                "Column": missing_only.index,
                "Missing": missing_only.values,
                "Missing %": (
                    missing_only.values
                    / len(quality_df)
                    * 100
                ),
            }
        )

        m1, m2 = st.columns(2)

        with m1:

            metric_card(
                "Columns with missing values",
                f"{len(missing_summary):,}",
            )

        with m2:

            metric_card(
                "Highest missingness",
                (
                    f"{missing_summary.iloc[0]['Missing %']:.1f}%"
                    if not missing_summary.empty
                    else "0%"
                ),
            )

        st.dataframe(
            missing_summary,
            use_container_width=True,
            hide_index=True,
        )

        fig, ax = plt.subplots(
            figsize=(9, 4.5)
        )

        plot_missing = (
            missing_summary
            .sort_values("Missing %")
        )

        ax.barh(
            plot_missing["Column"],
            plot_missing["Missing %"],
        )

        ax.set_xlabel(
            "Missing values (%)"
        )

        ax.set_title(
            f"Missingness in {quality_dataset.lower()}"
        )

        plt.tight_layout()

        st.pyplot(fig)

        figure_download_button(
            fig,
            "superlender_missingness.png",
        )

        plt.close(fig)

    # ========================================================
    # DUPLICATES & KEY INTEGRITY
    # ========================================================

    st.markdown("### Duplicate & key integrity")

    if quality_dataset == "Current applications":

        key_columns = [
            "systemloanid",
            "customerid",
        ]

    elif quality_dataset == "Historical loans":

        key_columns = [
            "systemloanid",
            "customerid",
        ]

    else:

        key_columns = [
            "customerid",
        ]

    duplicate_table = []

    for column in key_columns:

        duplicate_rows = quality_df[column].duplicated(
            keep=False
        )

        duplicate_table.append(
            {
                "Column": column,
                "Unique values": int(
                    quality_df[column].nunique(
                        dropna=True
                    )
                ),
                "Duplicate key values": int(
                    quality_df.loc[
                        duplicate_rows,
                        column,
                    ]
                    .dropna()
                    .nunique()
                ),
                "Rows involved": int(
                    duplicate_rows.sum()
                ),
            }
        )

    duplicate_result = pd.DataFrame(
        duplicate_table
    )

    st.dataframe(
        duplicate_result,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Duplicate keys are interpreted according to dataset grain. "
        "Repeated customer IDs are expected in historical loans but "
        "not for the current-loan prediction grain."
    )

    # ========================================================
    # NUMERICAL OUTLIERS
    # ========================================================

    st.markdown("### Numerical outliers")

    if quality_dataset == "Current applications":

        outlier_columns = [
            "loanamount",
            "totaldue",
        ]

    elif quality_dataset == "Historical loans":

        outlier_columns = [
            "loanamount",
            "totaldue",
            "loannumber",
        ]

    else:

        outlier_columns = []

    if quality_dataset == "Demographics":

        numeric_candidates = [
            column
            for column in quality_df.select_dtypes(
                include="number"
            ).columns
            if column not in [
                "customerid"
            ]
        ]

        outlier_columns = numeric_candidates

    outlier_result = calculate_iqr_outliers(
        quality_df,
        outlier_columns,
    )

    if outlier_result.empty:

        st.info(
            "No suitable numerical variables are available "
            "for IQR outlier diagnostics."
        )

    else:

        st.dataframe(
            outlier_result,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "IQR flags indicate unusual observations for statistical "
            "review. They do not mean the records are erroneous and "
            "are not automatically removed."
        )

    # ========================================================
    # LOGICAL CONSISTENCY
    # ========================================================

    st.markdown("### Logical consistency")

    if quality_dataset == "Current applications":

        consistency_result = (
            check_current_data_consistency(
                performance
            )
        )

    elif quality_dataset == "Historical loans":

        consistency_result = (
            check_historical_data_consistency(
                previous_loans
            )
        )

    else:

        consistency_result = pd.DataFrame(
            [
                {
                    "Check": "customerid uniqueness",
                    "Violations": int(
                        demographics[
                            "customerid"
                        ].duplicated().sum()
                    ),
                    "Expected": 0,
                },
            ]
        )

    consistency_display = (
        consistency_result.copy()
    )

    consistency_display["Status"] = (
        consistency_display["Violations"]
        .apply(quality_status)
    )

    st.dataframe(
        consistency_display[
            [
                "Check",
                "Violations",
                "Expected",
                "Status",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # GPS DIAGNOSTIC
    # ========================================================

    if quality_dataset == "Demographics":

        st.markdown("### Geographic diagnostic")

        gps_anomalies = find_gps_anomalies(
            demographics.rename(
                columns={
                    "latitude_gps": "latitude",
                    "longitude_gps": "longitude",
                }
            )
        )

        if gps_anomalies.empty:

            st.success(
                "No GPS coordinates were flagged by the diagnostic rule."
            )

        else:

            st.warning(
                f"{len(gps_anomalies):,} demographic records contain "
                "GPS coordinates flagged for review by the broad "
                "diagnostic range."
            )

            st.dataframe(
                gps_anomalies.head(50),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "These coordinates were excluded from the production "
                "feature set. The diagnostic highlights data-quality "
                "concerns; it does not determine the true geographic "
                "location of a customer."
            )

    # ========================================================
    # QUALITY INTERPRETATION
    # ========================================================

    st.markdown("### How to interpret these diagnostics")

    q_left, q_right = st.columns(2)

    with q_left:

        st.markdown("#### Statistical flags")

        st.write(
            "Missingness and IQR-based outlier detection identify "
            "patterns that require investigation. They do not imply "
            "that the underlying records are invalid."
        )

    with q_right:

        st.markdown("#### Modelling decisions")

        st.write(
            "Production preprocessing handles missing information "
            "explicitly, while suspicious fields were excluded only "
            "when there was a documented data-quality or availability "
            "reason."
        )


    # ========================================================
    # DATA QUALITY TAKEAWAYS
    # ========================================================

    section_header(
        "What the data quality checks show",
        "These are observations from the source datasets, not causal conclusions.",
    )

    q1, q2 = st.columns(2)

    with q1:

        st.markdown("#### Demographic coverage")

        st.write(
            "The current applications contain demographic matches "
            "for 3,269 applications, while 1,099 current applications "
            "do not have a matching training demographic record."
        )

        st.caption(
            "The unmatched records remain missing rather than being "
            "filled from the test demographic dataset."
        )

    with q2:

        st.markdown("#### Historical coverage")

        st.write(
            "Historical loan records exist for most current applications, "
            "while a small group has no matching previous-loan history."
        )

        st.caption(
            "Absence of historical records is treated as a meaningful "
            "data condition rather than evidence that the customer had "
            "never borrowed outside the available dataset."
        )

    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown(
        """
        <div class="footer">
            SuperLender • Data Understanding & Exploratory Analysis
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# MODEL TRAINING
# ============================================================

elif page == "Model Training":

    st.markdown(
        '<div class="brand-title">Model Training</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="brand-subtitle">'
        'Model comparison and intelligent selection'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Train multiple classification candidates on the same "
        "application-time feature set, compare their validation "
        "performance, and automatically identify the preferred "
        "model using the selected validation criterion."
    )

    st.info(
        "Bad is treated as the positive risk event. "
        "The final holdout remains outside this training workflow."
    )

    # ========================================================
    # TRAINING DATA
    # ========================================================

    @st.cache_data
    def load_training_frame(
        performance_df,
        previous_loans_df,
        demographics_df,
    ):

        return prepare_training_frame(
            performance_df,
            previous_loans_df,
            demographics_df,
        )

    training_frame = load_training_frame(
        performance,
        previous_loans,
        demographics,
    )

    numeric_features = FeatureEngineer.NUMERIC_FEATURES.copy()
    categorical_features = (
        FeatureEngineer.CATEGORICAL_FEATURES.copy()
    )

    feature_columns = (
        numeric_features
        + categorical_features
    )

    # ========================================================
    # TARGET / FEATURES
    # ========================================================

    section_header(
        "Training configuration",
        "The project target is good_bad_flag. "
        "Production modelling uses the canonical 34-feature contract.",
    )

    target_column = st.selectbox(
        "Target",
        ["good_bad_flag"],
        index=0,
        key="training_target",
    )

    selected_training_features = st.multiselect(
        "Features",
        feature_columns,
        default=feature_columns,
        key="training_features",
    )

    if not selected_training_features:

        st.warning(
            "Select at least one feature before training."
        )

        st.stop()

    selected_numeric = [
        feature
        for feature in numeric_features
        if feature in selected_training_features
    ]

    selected_categorical = [
        feature
        for feature in categorical_features
        if feature in selected_training_features
    ]

    # ========================================================
    # TEMPORAL SPLIT
    # ========================================================

    section_header(
        "Validation strategy",
        "Applications are ordered chronologically so that validation "
        "represents later applications than the training period.",
    )

    train_df, validation_df, final_holdout_df = (
        chronological_train_validation_split(
            training_frame
        )
    )

    x_train = train_df[
        selected_training_features
    ]

    y_train = (
        train_df[target_column]
        .map(
            {
                "Good": 1,
                "Bad": 0,
            }
        )
    )

    x_validation = validation_df[
        selected_training_features
    ]

    y_validation = (
        validation_df[target_column]
        .map(
            {
                "Good": 1,
                "Bad": 0,
            }
        )
    )

    split1, split2, split3, split4 = st.columns(4)

    with split1:
        metric_card(
            "Training rows",
            f"{len(train_df):,}",
            "Chronological first 70%",
        )

    with split2:
        metric_card(
            "Validation rows",
            f"{len(validation_df):,}",
            "Later applications",
        )

    with split3:
        metric_card(
            "Training Bad rate",
            f"{(y_train == 0).mean():.1%}",
        )

    with split4:
        metric_card(
            "Validation Bad rate",
            f"{(y_validation == 0).mean():.1%}",
        )

    st.caption(
        f"Training period: "
        f"{train_df['creationdate'].min():%d %b %Y %H:%M}"
        f" → "
        f"{train_df['creationdate'].max():%d %b %Y %H:%M}"
    )

    st.caption(
        f"Validation period: "
        f"{validation_df['creationdate'].min():%d %b %Y %H:%M}"
        f" → "
        f"{validation_df['creationdate'].max():%d %b %Y %H:%M}"
    )

    # ========================================================
    # SELECTION CRITERION
    # ========================================================

    section_header(
        "Model selection criterion",
        "The preferred candidate is identified automatically from "
        "validation performance rather than by model name.",
    )

    criterion = st.selectbox(
        "Primary criterion",
        [
            "PR-AUC",
            "F1-score",
            "ROC-AUC",
        ],
        index=0,
        key="training_selection_criterion",
    )
    
    criterion_display = {
        "PR-AUC": "Validation PR-AUC",
        "F1-score": "Validation F1",
        "ROC-AUC": "Validation ROC-AUC",
    }[criterion]

    criterion_mapping = {
        "PR-AUC": "pr_auc",
        "F1-score": "f1_bad",
        "ROC-AUC": "roc_auc",
    }

    criterion_column = criterion_mapping[
        criterion
    ]

    st.caption(
        f"Current selection criterion: **{criterion_display}**, "
        "with Bad treated as the positive risk event."
    )

    # ========================================================
    # TRAIN MODELS
    # ========================================================

    section_header(
        "Candidate models",
        "Each candidate uses the same selected features, "
        "missing-value handling and chronological validation boundary.",
    )

    if st.button(
        "Train and compare models",
        type="primary",
        use_container_width=False,
    ):

        candidates = build_candidate_models(
            selected_numeric,
            selected_categorical,
        )

        results = []
        trained_models = {}

        progress = st.progress(
            0
        )

        status = st.empty()

        for index, (
            model_name,
            pipeline,
        ) in enumerate(
            candidates.items(),
            start=1,
        ):

            status.write(
                f"Training **{model_name}**..."
            )

            pipeline.fit(
                x_train,
                y_train,
            )

            metrics = evaluate_bad_risk_model(
                pipeline,
                x_validation,
                y_validation,
            )

            trained_models[
                model_name
            ] = pipeline

            results.append(
                {
                    "Model": model_name,
                    "Accuracy": metrics["accuracy"],
                    "Precision (Bad)": metrics[
                        "precision_bad"
                    ],
                    "Recall (Bad)": metrics[
                        "recall_bad"
                    ],
                    "F1 (Bad)": metrics[
                        "f1_bad"
                    ],
                    "ROC-AUC": metrics[
                        "roc_auc"
                    ],
                    "PR-AUC": metrics[
                        "pr_auc"
                    ],
                    "Average Precision": metrics[
                        "average_precision"
                    ],
                    "Confusion Matrix": metrics[
                        "confusion_matrix"
                    ],
                }
            )

            progress.progress(
                index / len(candidates)
            )

        status.empty()
        progress.empty()

        results_df = pd.DataFrame(
            results
        )

        # ----------------------------------------------------
        # Automatic selection
        # ----------------------------------------------------

        results_df = results_df.sort_values(
            by=[
                {
                    "PR-AUC": "PR-AUC",
                    "F1-score": "F1 (Bad)",
                    "ROC-AUC": "ROC-AUC",
                }[criterion],
                "F1 (Bad)",
            ],
            ascending=False,
        ).reset_index(
            drop=True
        )

        preferred_model = results_df.iloc[
            0
        ]["Model"]

        st.session_state[
            "training_results"
        ] = results_df

        st.session_state[
            "trained_candidate_models"
        ] = trained_models

        st.session_state[
            "preferred_training_model"
        ] = preferred_model

        # Freeze the exact fitted model, feature list and selection
        # settings used for this training run.
        st.session_state["training_model"] = trained_models[preferred_model]
        st.session_state["training_model_name"] = preferred_model
        st.session_state["training_model_features"] = selected_training_features.copy()
        st.session_state["training_model_threshold"] = 0.50
        st.session_state["training_model_selection_criterion"] = criterion
        st.session_state["training_model_run_id"] += 1

        # Any batch scored with an earlier training-session model is stale.
        st.session_state.pop("batch_results", None)
        st.session_state.pop("batch_results_mode", None)
        st.session_state.pop("batch_results_upload_signature", None)

    # ========================================================
    # RESULTS
    # ========================================================

    if (
        "training_results"
        in st.session_state
    ):

        results_df = st.session_state[
            "training_results"
        ]

        preferred_model = st.session_state[
            "preferred_training_model"
        ]

        st.success(
            f"Preferred model identified automatically: "
            f"**{preferred_model}**"
        )

        display_results = results_df[
            [
                "Model",
                "Accuracy",
                "Precision (Bad)",
                "Recall (Bad)",
                "F1 (Bad)",
                "ROC-AUC",
                "PR-AUC",
            ]
        ].copy()

        percentage_columns = [
            "Accuracy",
            "Precision (Bad)",
            "Recall (Bad)",
        ]

        for column in percentage_columns:
            display_results[
                column
            ] = (
                display_results[
                    column
                ] * 100
            ).round(1)

        display_results[
            "F1 (Bad)"
        ] = display_results[
            "F1 (Bad)"
        ].round(3)

        display_results[
            "ROC-AUC"
        ] = display_results[
            "ROC-AUC"
        ].round(3)

        display_results[
            "PR-AUC"
        ] = display_results[
            "PR-AUC"
        ].round(3)

        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True,
        )

        if st.session_state.get("training_model") is not None:
            trained_criterion = st.session_state.get(
                "training_model_selection_criterion",
                criterion,
            )
            if trained_criterion is None:
                trained_criterion = "PR-AUC"

            trained_criterion_display = {
                "PR-AUC": "Validation PR-AUC",
                "F1-score": "Validation F1",
                "ROC-AUC": "Validation ROC-AUC",
            }.get(trained_criterion, "Validation PR-AUC")

            st.write(
                f"The table is ordered by **{trained_criterion_display}**. "
                "The primary selection criterion determines the preferred "
                "candidate; F1 is used as a secondary tie-breaker."
            )


        if st.button("Clear training selection"):
            st.session_state["training_model"] = None
            st.session_state["training_model_name"] = None
            st.session_state["training_model_features"] = None
            st.session_state["training_model_threshold"] = 0.50
            st.session_state["training_model_selection_criterion"] = "PR-AUC"
            st.session_state["risk_inference_mode"] = "Production model"
            st.session_state["batch_inference_mode"] = "Production model"
            st.rerun()

        # ====================================================
        # SELECTED MODEL
        # ====================================================

        trained_criterion = st.session_state.get(
                "training_model_selection_criterion",
                criterion,
            )
        if trained_criterion is None:
            trained_criterion = "PR-AUC"

        trained_criterion_display = {
            "PR-AUC": "Validation PR-AUC",
            "F1-score": "Validation F1",
            "ROC-AUC": "Validation ROC-AUC",
        }.get(trained_criterion, "Validation PR-AUC")

        section_header(
            "Preferred model",
            "Automatically selected from the validation comparison.",
        )

        st.caption(
            f"Ranking criterion: {trained_criterion_display}. "
            "Classification metrics below use the 50% reference threshold."
        )

        selected_row = results_df[
            results_df["Model"]
            == preferred_model
        ].iloc[0]

        p1, p2, p3, p4 = st.columns(4)

        with p1:
            metric_card(
                "Model",
                preferred_model,
                "Validation selection",
            )

        with p2:
            selected_criterion_value = selected_row[
                {
                    "PR-AUC": "PR-AUC",
                    "F1-score": "F1 (Bad)",
                    "ROC-AUC": "ROC-AUC",
                }[trained_criterion]
            ]

            metric_card(
                trained_criterion_display,
                f"{selected_criterion_value:.3f}",
                f"Bad-event ranking • selected criterion: {trained_criterion}",
            )

        with p3:
            metric_card(
                "Bad recall",
                f"{selected_row['Recall (Bad)']:.1%}",
                "Share of observed Bad loans detected",
            )

        with p4:
            metric_card(
                "Bad F1",
                f"{selected_row['F1 (Bad)']:.3f}",
                "Precision/recall balance at 50%",
            )

        st.info(
            f"{preferred_model} is currently selected because it has "
            f"the highest {trained_criterion_display} on the chronological validation set. "
            "This selection is based on validation performance, not training "
            "accuracy or model name."
        )

        # ====================================================
        # CONFUSION MATRICES
        # ====================================================

        section_header(
            "Classification errors",
            "Confusion matrices show how each candidate converts "
            "risk scores into classifications at the 50% reference threshold.",
        )

        st.caption(
            "Reference classification threshold: 50%. "
            "Bad is treated as the positive risk event."
        )

        matrix_columns = st.columns(
            len(results_df)
        )

        for column, (_, row) in zip(
            matrix_columns,
            results_df.iterrows(),
        ):

            with column:

                st.markdown(
                    f"**{row['Model']}**"
                )

                matrix = np.array(
                    row["Confusion Matrix"]
                )

                matrix_display = pd.DataFrame(
                    matrix,
                    index=[
                        "Actual Bad",
                        "Actual Good",
                    ],
                    columns=[
                        "Predicted Bad",
                        "Predicted Good",
                    ],
                )

                st.dataframe(
                    matrix_display,
                    use_container_width=True,
                )
        st.caption(
            "TP = correctly identified Bad loans; FN = Bad loans classified Good; "
            "FP = Good loans classified Bad; TN = correctly identified Good loans."
        )

        st.markdown("#### Final holdout protection")

        holdout_col1, holdout_col2, holdout_col3 = st.columns(3)

        with holdout_col1:
            metric_card(
                "Final holdout",
                f"{len(final_holdout_df):,}",
                "15% reserved for final evaluation",
            )

        with holdout_col2:
            holdout_start = pd.to_datetime(
                final_holdout_df["creationdate"]
            ).min()

            holdout_end = pd.to_datetime(
                final_holdout_df["creationdate"]
            ).max()

            metric_card(
                "Holdout period",
                f"{holdout_start:%d %b} → {holdout_end:%d %b}",
                "Later applications",
            )

        with holdout_col3:
            metric_card(
                "Model selection",
                "Excluded",
                "Not used for training or selection",
            )

        st.info(
            "The final holdout is kept completely outside candidate-model "
            "training and selection. It is used only for the final evaluation "
            "after the model-selection decision is frozen."
        )

    # ========================================================
    # PRODUCTION MODEL NOTICE
    # ========================================================

    section_header(
        "Production model status",
        "Training experiments and production inference are kept separate.",
    )

    st.write(
        "The production inference pipeline currently uses the "
        "frozen HistGradientBoosting production artifact and its "
        "configured 23% operating threshold."
    )

    st.caption(
        "A training experiment does not overwrite the saved production artifact. "
        "Production changes require an explicit promotion and artifact-update step."
    )

# ============================================================
# RISK ASSESSMENT
# ============================================================

elif page == "Risk Assessment":

    st.markdown(
        '<div class="brand-title">Risk Assessment</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="brand-subtitle">'
        'Application-level credit-risk assessment'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Select an existing application from the training dataset "
        "to score it using the selected SuperLender feature-engineering "
        "and inference model."
    )


    # ========================================================
    # APPLICATION SELECTION
    # ========================================================

    selected_loan_id = st.selectbox(
        "Select a current application",
        options=performance["systemloanid"].tolist(),
        format_func=lambda loan_id: f"Loan {loan_id}",
    )


    selected_current = (
        performance[
            performance["systemloanid"]
            == selected_loan_id
        ]
        .copy()
    )


    if selected_current.empty:

        st.error(
            "The selected application could not be found."
        )

        st.stop()


    customer_id = (
        selected_current["customerid"]
        .iloc[0]
    )


    selected_history = (
        previous_loans[
            previous_loans["customerid"]
            == customer_id
        ]
        .copy()
    )


    selected_demographics = (
        demographics[
            demographics["customerid"]
            == customer_id
        ]
        .copy()
    )


    row = selected_current.iloc[0]


    # ========================================================
    # APPLICATION PROFILE
    # ========================================================

    section_header(
        "Application profile",
        "Current-loan information used by the selected inference pipeline.",
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        metric_card(
            "Loan amount",
            f"₦{row['loanamount']:,.0f}",
            "Current request",
        )


    with c2:

        metric_card(
            "Total due",
            f"₦{row['totaldue']:,.0f}",
            "Current obligation",
        )


    with c3:

        metric_card(
            "Loan term",
            f"{int(row['termdays'])} days",
            "Current term",
        )


    with c4:

        metric_card(
            "Loan number",
            f"{int(row['loannumber'])}",
            "Repeat borrowing position",
        )


    # ========================================================
    # APPLICATION CONTEXT
    # ========================================================

    left, right = st.columns(2)


    with left:

        st.markdown("#### Application")

        st.write(
            f"**Loan ID:** `{selected_loan_id}`"
        )

        st.write(
            f"**Customer ID:** `{customer_id}`"
        )

        st.write(
            f"**Created:** "
            f"{pd.to_datetime(row['creationdate']):%d %b %Y, %H:%M}"
        )


    with right:

        st.markdown("#### Available customer evidence")

        st.write(
            f"**Historical loans:** "
            f"{len(selected_history):,}"
        )

        st.write(
            f"**Demographic record:** "
            f"{'Available' if not selected_demographics.empty else 'Unavailable'}"
        )


    # ========================================================
    # INFERENCE MODEL
    # ========================================================

    training_model_available = (
        st.session_state.get("training_model") is not None
    )

    inference_options = [
        "Production model",
        "Current training selection",
    ]

    if (
        st.session_state.get("risk_inference_mode")
        == "Current training selection"
        and not training_model_available
    ):
        st.session_state["risk_inference_mode"] = "Production model"

    inference_mode = st.radio(
        "Inference model",
        inference_options,
        horizontal=True,
        key="risk_inference_mode",
    )

    if not training_model_available:
        st.caption(
            "Current training selection is available after a model is "
            "trained and selected on the Model Training page."
        )

    if (
        inference_mode == "Current training selection"
        and not training_model_available
    ):
        st.warning(
            "No current Streamlit training model is available. "
            "Train and select a model on the Model Training page first."
        )
        st.stop()

    if inference_mode == "Production model":

        risk_description = (
            "The application is scored using the frozen production "
            "model and its configured operating threshold. "
            "This is the default inference path."
        )

        try:
            prediction = risk_service.predict_single(
                current_loan=row.drop(labels=["good_bad_flag"], errors="ignore").to_dict(),
                previous_loans=selected_history,
                demographics=selected_demographics,
            )

        except Exception as exc:
            st.error(
                "The production prediction pipeline could not "
                "score this application."
            )
            st.exception(exc)
            st.stop()

        if isinstance(prediction, pd.DataFrame):
            result = prediction.iloc[0]
        else:
            result = prediction

        bad_risk_score = float(result["bad_risk_score"])
        decision_threshold = float(result["decision_threshold"])
        predicted_class = str(result["predicted_class"])
        model_name = str(result["model_name"])
        model_version = str(result["model_version"])
        active_model_label = model_name
        active_mode_label = "Production model"

    else:

        risk_description = (
            "The application is scored using the model selected in the "
            "current Streamlit training session. The 50% reference "
            "threshold is used because no candidate-specific production "
            "threshold has been promoted."
        )

        selected_pipeline = st.session_state.get("training_model")
        selected_model_name = st.session_state.get("training_model_name")
        selected_features = st.session_state.get("training_model_features")
        selected_threshold = float(
            st.session_state.get("training_model_threshold", 0.50)
        )

        if (
            selected_pipeline is None
            or not selected_model_name
            or not selected_features
        ):
            st.warning(
                "No current Streamlit training selection is available. "
                "Train and select a model first."
            )
            st.stop()

        try:
            training_feature_engineer = FeatureEngineer()

            training_engineered = training_feature_engineer.build_features(
                current_loans=selected_current,
                previous_loans=selected_history,
                demographics=selected_demographics,
            )

            predicted_class, bad_risk_score = predict_with_streamlit_model(
                pipeline=selected_pipeline,
                engineered_features=training_engineered,
                feature_names=selected_features,
                threshold=selected_threshold,
            )

        except Exception as exc:
            st.error(
                "The current Streamlit training model could not "
                "score this application."
            )
            st.exception(exc)
            st.stop()

        decision_threshold = selected_threshold
        model_name = selected_model_name
        model_version = "streamlit-session"
        active_model_label = selected_model_name
        active_mode_label = "Current training selection"

    section_header(
        "Risk estimate",
        risk_description,
    )

    st.caption(
        f"Active inference mode: **{active_mode_label}**  "
        f"| Model: **{active_model_label}**  "
        f"| Classification threshold: **{decision_threshold:.0%}**"
    )

    # ========================================================
    # RISK RESULT + MODEL CONTEXT
    # ========================================================

    left, right = st.columns([1.65, 1])

    with left:
        display_risk_result(
            bad_risk_score=bad_risk_score,
            threshold=decision_threshold,
            predicted_class=predicted_class,
        )

    with right:
        st.markdown("#### Model context")

        metric_card(
            "Inference mode",
            active_mode_label,
            f"{model_name} • Version {model_version}",
        )

        st.caption(f"Technical estimator: {model_name}")

        st.markdown("")
        st.caption("Classification threshold")
        st.markdown(f"### {decision_threshold:.1%}")

        threshold_caption = (
            "Configured production decision threshold"
            if inference_mode == "Production model"
            else "50% reference threshold for training-session inference"
        )

        st.caption(threshold_caption)

    # ========================================================
    # ENGINEERED BEHAVIOURAL EVIDENCE
    # ========================================================

    evidence_description = (
        "Historical information observable before the current "
        "application and used by the production feature pipeline."
        if inference_mode == "Production model"
        else
        "Historical information observable before the current "
        "application and used by the same FeatureEngineer during "
        "training-session inference."
    )

    section_header(
        "Behavioural evidence",
        evidence_description,
    )

    try:
        feature_engineer = FeatureEngineer()

        engineered = feature_engineer.build_features(
            current_loans=selected_current,
            previous_loans=selected_history,
            demographics=selected_demographics,
        )

    except Exception as exc:
        st.warning(
            "The prediction succeeded, but the behavioural "
            "evidence panel could not be generated."
        )
        st.exception(exc)
        engineered = pd.DataFrame()




    if not engineered.empty:

        engineered_row = engineered.iloc[0]


        # ----------------------------------------------------
        # Extract behavioural features
        # ----------------------------------------------------

        historical_count = engineered_row.get(
            "historical_loan_count"
        )

        observed_count = engineered_row.get(
            "observed_repayment_count"
        )

        late_count = engineered_row.get(
            "observed_late_repayment_count"
        )

        late_rate = engineered_row.get(
            "observed_late_repayment_rate"
        )

        avg_delay = engineered_row.get(
            "avg_observed_repayment_delay_days"
        )

        recency = engineered_row.get(
            "days_since_last_loan"
        )


        # ----------------------------------------------------
        # Human-readable values
        # ----------------------------------------------------

        history_text = (
            "No history"
            if pd.isna(historical_count)
            else f"{int(historical_count)}"
        )


        observed_text = (
            "None observed"
            if pd.isna(observed_count)
            else f"{int(observed_count)}"
        )


        late_count_text = (
            "None"
            if pd.isna(late_count)
            else f"{int(late_count)}"
        )


        late_rate_text = (
            "No observed repayments"
            if pd.isna(late_rate)
            else f"{late_rate:.1%}"
        )


        delay_text = (
            "No observed repayments"
            if pd.isna(avg_delay)
            else (
                f"{abs(avg_delay):.1f} days early"
                if avg_delay < 0
                else (
                    "On time"
                    if avg_delay == 0
                    else f"{avg_delay:.1f} days late"
                )
            )
        )


        recency_text = (
            "No history"
            if pd.isna(recency)
            else f"{recency:.1f} days"
        )


        # ----------------------------------------------------
        # Evidence cards
        # ----------------------------------------------------

        e1, e2, e3 = st.columns(3)


        with e1:

            metric_card(
                "Historical loans",
                history_text,
                "Observable before application",
            )


        with e2:

            metric_card(
                "Observed repayments",
                observed_text,
                "Repayment events available at prediction time",
            )


        with e3:

            metric_card(
                "Late repayments",
                late_count_text,
                "Among observed historical repayments",
            )


        e4, e5, e6 = st.columns(3)


        with e4:

            metric_card(
                "Historical late rate",
                late_rate_text,
                "Observed historical repayments",
            )


        with e5:

            metric_card(
                "Average repayment behaviour",
                delay_text,
                "Across observed repayments",
            )


        with e6:

            metric_card(
                "Days since last loan",
                recency_text,
                "Historical approval → current application",
            )


        st.caption(
            "These features describe evidence available at application "
            "time. They should be interpreted as model inputs or "
            "associations, not as proof that any individual feature "
            "caused the outcome."
        )


    # ========================================================
    # RETROSPECTIVE VALIDATION CONTEXT
    # ========================================================

    with st.expander(
        "Retrospective outcome for this historical application"
    ):

        actual_class = str(
            row["good_bad_flag"]
        )

        st.write(
            "Because this is a historical training application, "
            "its observed outcome is known. This information is "
            "shown only for retrospective model inspection and "
            "is not used as an input to the prediction."
        )

        st.metric(
            "Observed outcome",
            actual_class,
        )


    # ========================================================
    # RAW DATA DETAILS
    # ========================================================

    with st.expander(
        "View application record"
    ):

        st.dataframe(
            selected_current,
            use_container_width=True,
            hide_index=True,
        )


    with st.expander(
        "View available historical records"
    ):

        if selected_history.empty:

            st.info(
                "No historical loan records are available "
                "for this customer."
            )

        else:

            st.dataframe(
                selected_history,
                use_container_width=True,
                hide_index=True,
            )

# ============================================================
# BATCH INTELLIGENCE
# ============================================================

elif page == "Batch Intelligence":

    st.markdown(
        '<div class="brand-title">Batch Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="brand-subtitle">'
        'Portfolio-level application risk scoring'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Upload a batch of current loan applications and "
        "SuperLender will score them using the same application-time "
        "feature-engineering pipeline used for single-application assessment. "
        "Choose the inference model below."
    )


    # ========================================================
    # REFERENCE DATA
    # ========================================================

    section_header(
        "Reference data",
        "Historical loans and demographic records remain "
        "reference tables for the uploaded current applications.",
    )


    r1, r2, r3 = st.columns(3)


    with r1:

        metric_card(
            "Current reference",
            f"{len(performance):,}",
            "Training applications",
        )


    with r2:

        metric_card(
            "Historical reference",
            f"{len(previous_loans):,}",
            "Previous-loan records",
        )


    with r3:

        metric_card(
            "Demographic reference",
            f"{demographics['customerid'].nunique():,}",
            "Unique customer IDs",
        )


    # ========================================================
    # FILE UPLOAD
    # ========================================================

    section_header(
        "Upload current applications",
        "The CSV should contain current-loan information. "
        "Historical loans and demographics are matched using "
        "customerid.",
    )


    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help=(
            "Upload current application records only. "
            "Do not include good_bad_flag."
        ),
    )


    if uploaded_file is None:

        st.info(
            "Upload a CSV to begin batch scoring."
        )

        st.stop()


    # ========================================================
    # LOAD UPLOAD
    # ========================================================

    try:

        uploaded_bytes = uploaded_file.getvalue()

        batch_df = pd.read_csv(
            uploaded_file
        )

    except Exception as exc:

        st.error(
            "The uploaded file could not be read as CSV."
        )

        st.exception(exc)

        st.stop()


    # ========================================================
    # TRACK UPLOAD VERSION
    # ========================================================

    upload_signature = hashlib.sha256(
        uploaded_bytes
    ).hexdigest()

    if (
        st.session_state.get("batch_upload_signature")
        != upload_signature
    ):
        st.session_state.pop("batch_results", None)
        st.session_state["batch_upload_signature"] = upload_signature

    # ========================================================
    # VALIDATE UPLOAD
    # ========================================================

    validation_errors = validate_batch_input(
        batch_df
    )


    if validation_errors:

        st.error(
            "The uploaded file failed validation."
        )

        for error in validation_errors:

            st.write(
                f"- {error}"
            )

        st.stop()


    # ========================================================
    # BATCH PREVIEW
    # ========================================================

    section_header(
        "Batch preview",
        "Validated current applications ready for batch scoring.",
    )


    p1, p2, p3, p4 = st.columns(4)


    with p1:

        metric_card(
            "Applications",
            f"{len(batch_df):,}",
            "Uploaded records",
        )


    with p2:

        matched_history_count = count_applications_with_observable_history(
            batch_df,
            previous_loans,
        )

        metric_card(
            "Applications with observable history",
            f"{matched_history_count:,}",
            f"{matched_history_count / len(batch_df):.1%} of batch",
        )


    with p3:

        matched_demographic_count = (
            batch_df["customerid"]
            .isin(
                demographics["customerid"]
            )
            .sum()
        )

        metric_card(
            "Demographic matches",
            f"{matched_demographic_count:,}",
            f"{matched_demographic_count / len(batch_df):.1%} of batch",
        )


    with p4:

        no_history_count = (
            len(batch_df)
            - matched_history_count
        )

        metric_card(
            "No observable history",
            f"{no_history_count:,}",
            "Handled by the feature pipeline",
        )


    with st.expander(
        "Preview uploaded applications"
    ):

        st.dataframe(
            batch_df.head(25),
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # INFERENCE MODEL
    # ========================================================

    training_model_available = (
        st.session_state.get("training_model") is not None
    )

    batch_inference_options = [
        "Production model",
        "Current training selection",
    ]

    if (
        st.session_state.get("batch_inference_mode")
        == "Current training selection"
        and not training_model_available
    ):
        st.session_state["batch_inference_mode"] = "Production model"

    batch_inference_mode = st.radio(
        "Inference model",
        batch_inference_options,
        horizontal=True,
        key="batch_inference_mode",
    )

    if not training_model_available:
        st.caption(
            "Current training selection is available after a model is "
            "trained and selected on the Model Training page."
        )

    previous_batch_mode = st.session_state.get(
        "last_batch_inference_mode"
    )

    if previous_batch_mode != batch_inference_mode:
        st.session_state.pop("batch_results", None)
        st.session_state["last_batch_inference_mode"] = batch_inference_mode

    batch_mode_caption = (
        "Scores use the frozen production pipeline and its configured "
        "operating threshold."
        if batch_inference_mode == "Production model"
        else
        "Scores use the model selected in the current Streamlit training "
        "session at its 50% reference threshold."
    )

    st.caption(batch_mode_caption)

    # ========================================================
    # SCORE BUTTON
    # ========================================================

    st.markdown("")

    score_batch = st.button(
        "Score batch",
        type="primary",
        use_container_width=False,
    )


    if score_batch:

        if (
            batch_inference_mode == "Current training selection"
            and not training_model_available
        ):
            st.warning(
                "No current Streamlit training model is available. "
                "Train and select a model on the Model Training page first."
            )
            st.stop()

        if batch_inference_mode == "Production model":

            with st.spinner("Running production risk pipeline..."):
                try:
                    batch_predictions = risk_service.predict_batch(
                        current_loans=batch_df.copy(),
                        previous_loans=previous_loans.copy(),
                        demographics=demographics.copy(),
                    )

                except Exception as exc:
                    st.error(
                        "The production batch prediction pipeline could "
                        "not score the uploaded data."
                    )
                    st.exception(exc)
                    st.stop()

            if isinstance(batch_predictions, pd.DataFrame):
                results = batch_predictions.copy()
            else:
                results = pd.DataFrame(batch_predictions)

        else:

            selected_pipeline = st.session_state.get("training_model")
            selected_model_name = st.session_state.get("training_model_name")
            selected_features = st.session_state.get("training_model_features")
            selected_threshold = float(
                st.session_state.get("training_model_threshold", 0.50)
            )

            if (
                selected_pipeline is None
                or not selected_model_name
                or not selected_features
            ):
                st.warning(
                    "No current Streamlit training selection is available. "
                    "Train and select a model first."
                )
                st.stop()

            with st.spinner("Running current training-model pipeline..."):
                try:
                    batch_feature_engineer = FeatureEngineer()

                    engineered_batch = batch_feature_engineer.build_features(
                        current_loans=batch_df.copy(),
                        previous_loans=previous_loans.copy(),
                        demographics=demographics.copy(),
                    )

                    predicted_classes, bad_scores = predict_batch_with_streamlit_model(
                        pipeline=selected_pipeline,
                        engineered_features=engineered_batch,
                        feature_names=selected_features,
                        threshold=selected_threshold,
                    )

                except Exception as exc:
                    st.error(
                        "The current Streamlit training model could "
                        "not score the uploaded batch."
                    )
                    st.exception(exc)
                    st.stop()

            results = batch_df[
                ["systemloanid", "customerid"]
            ].copy()

            results["bad_risk_score"] = bad_scores
            results["decision_threshold"] = selected_threshold
            results["predicted_class"] = predicted_classes
            results["model_name"] = selected_model_name
            results["model_version"] = "streamlit-session"

        # ----------------------------------------------------
        # Integrity check
        # ----------------------------------------------------

        if len(results) != len(batch_df):
            st.error(
                "The prediction pipeline did not return exactly one "
                "result per uploaded application."
            )

            st.write(f"Uploaded rows: {len(batch_df):,}")
            st.write(f"Prediction rows: {len(results):,}")
            st.stop()

        required_result_columns = {
            "systemloanid",
            "customerid",
            "bad_risk_score",
            "decision_threshold",
            "predicted_class",
            "model_name",
            "model_version",
        }

        missing_result_columns = (
            required_result_columns - set(results.columns)
        )

        if missing_result_columns:
            st.error(
                "The prediction service returned an incomplete result. "
                f"Missing columns: {sorted(missing_result_columns)}"
            )
            st.stop()

        st.session_state["batch_results"] = results
        st.session_state["batch_results_mode"] = batch_inference_mode
        st.session_state["batch_results_upload_signature"] = upload_signature

    # ========================================================
    # DISPLAY STORED RESULTS
    # ========================================================

    results_are_current = (
        "batch_results" in st.session_state
        and st.session_state.get("batch_results_mode") == batch_inference_mode
        and st.session_state.get("batch_results_upload_signature") == upload_signature
    )

    if results_are_current:

        results = st.session_state[
            "batch_results"
        ].copy()


        # ====================================================
        # RESULT SUMMARY
        # ====================================================

        result_mode = st.session_state.get(
            "batch_results_mode",
            "Production model",
        )

        if result_mode == "Production model":
            summary_description = (
                "Results from the frozen production model and its "
                "configured operating threshold."
            )
            scored_caption = "Production predictions"
        else:
            summary_description = (
                "Results from the model selected in the current Streamlit "
                "training session using the 50% reference threshold."
            )
            scored_caption = "Training-session predictions"

        section_header(
            "Batch risk summary",
            summary_description,
        )


        total_scored = len(results)

        bad_predictions = (
            results["predicted_class"]
            == "Bad"
        ).sum()

        good_predictions = (
            results["predicted_class"]
            == "Good"
        ).sum()

        average_risk = (
            results["bad_risk_score"]
            .mean()
        )


        s1, s2, s3, s4 = st.columns(4)


        with s1:

            metric_card(
                "Applications scored",
                f"{total_scored:,}",
                scored_caption,
            )


        with s2:

            metric_card(
                "Estimated Bad",
                f"{bad_predictions:,}",
                f"{bad_predictions / total_scored:.1%} of batch",
            )


        with s3:

            metric_card(
                "Estimated Good",
                f"{good_predictions:,}",
                f"{good_predictions / total_scored:.1%} of batch",
            )


        with s4:

            metric_card(
                "Average Bad-risk score",
                f"{average_risk:.1%}",
                "Across uploaded applications",
            )


        # ====================================================
        # RISK DISTRIBUTION
        # ====================================================

        left, right = st.columns(
            [1.5, 1]
        )


        with left:

            fig, ax = plt.subplots(
                figsize=(8, 4)
            )

            ax.hist(
                results["bad_risk_score"],
                bins=20,
            )

            ax.axvline(
                results["decision_threshold"].iloc[0],
                linestyle="--",
                linewidth=2,
            )

            ax.set_xlabel(
                "Estimated Bad Risk"
            )

            ax.set_ylabel(
                "Number of applications"
            )

            ax.set_title(
                "Distribution of estimated Bad-risk scores"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


        with right:

            st.markdown(
                "### Decision boundary"
            )

            threshold = float(
                results[
                    "decision_threshold"
                ].iloc[0]
            )

            st.markdown(
                f"## {threshold:.1%}"
            )

            st.write(
                "Applications with scores at or above the configured "
                "threshold are classified as Bad by the active "
                "classification rule."
            )

            st.caption(
                "The threshold is a decision rule, not "
                "a statement that the score is perfectly calibrated."
            )


        # ====================================================
        # RESULTS TABLE
        # ====================================================

        section_header(
            "Application-level results",
            "Sort and inspect the model's risk estimates and "
            "resulting classifications.",
        )


        display_columns = [
            column
            for column in [
                "systemloanid",
                "customerid",
                "bad_risk_score",
                "decision_threshold",
                "predicted_class",
                "model_name",
                "model_version",
            ]
            if column in results.columns
        ]


        display_results = (
            results[
                display_columns
            ]
            .copy()
        )


        if "bad_risk_score" in display_results.columns:

            display_results[
                "bad_risk_score"
            ] = (
                display_results[
                    "bad_risk_score"
                ] * 100
            ).round(2)


        if "decision_threshold" in display_results.columns:

            display_results[
                "decision_threshold"
            ] = (
                display_results[
                    "decision_threshold"
                ] * 100
            ).round(2)


        display_results = display_results.rename(
            columns={
                "systemloanid": "Loan ID",
                "customerid": "Customer ID",
                "bad_risk_score": "Bad Risk (%)",
                "decision_threshold": "Threshold (%)",
                "predicted_class": "Decision",
                "model_name": "Model",
                "model_version": "Version",
            }
        )


        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True,
        )


        # ====================================================
        # HIGHER-RISK FILTER
        # ====================================================

        threshold = float(
            results[
                "decision_threshold"
            ].iloc[0]
        )


        high_risk_results = (
            results[
                results["bad_risk_score"]
                >= threshold
            ]
            .copy()
        )


        with st.expander(
            "View applications at or above the current threshold"
        ):

            st.write(
                f"{len(high_risk_results):,} applications "
                f"are at or above the {threshold:.1%} "
                f"operating threshold."
            )

            if high_risk_results.empty:

                st.info(
                    "No applications in this batch are at or above "
                    "the current threshold."
                )

            else:

                high_risk_display = (
                    high_risk_results[
                        display_columns
                    ]
                    .copy()
                )

                if (
                    "bad_risk_score"
                    in high_risk_display.columns
                ):

                    high_risk_display[
                        "bad_risk_score"
                    ] = (
                        high_risk_display[
                            "bad_risk_score"
                        ] * 100
                    ).round(2)

                high_risk_display = (
                    high_risk_display.rename(
                        columns={
                            "systemloanid": "Loan ID",
                            "customerid": "Customer ID",
                            "bad_risk_score": "Bad Risk (%)",
                            "decision_threshold": "Threshold",
                            "predicted_class": "Decision",
                            "model_name": "Model",
                            "model_version": "Version",
                        }
                    )
                )

                st.dataframe(
                    high_risk_display,
                    use_container_width=True,
                    hide_index=True,
                )


        # ====================================================
        # DOWNLOAD RESULTS
        # ====================================================

        section_header(
            "Export",
            "Download the batch predictions for downstream "
            "review or analysis.",
        )


        download_df = results.copy()


        csv_data = download_df.to_csv(
            index=False
        ).encode("utf-8")


        st.download_button(
            label="Download batch results as CSV",
            data=csv_data,
            file_name="superlender_batch_predictions.csv",
            mime="text/csv",
        )


        st.caption(
            "Batch predictions are model estimates generated "
            "from the uploaded applications and application-time "
            "reference data."
        )

# ============================================================
# MODEL & EVIDENCE
# ============================================================

elif page == "Model & Evidence":

    st.markdown(
        '<div class="brand-title">Model & Evidence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="brand-subtitle">'
        'Validation, decision threshold and model evidence'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Inspect the evidence behind the production risk model, "
        "including validation performance, threshold behaviour, "
        "feature importance and known limitations."
    )


    # ========================================================
    # PRODUCTION MODEL
    # ========================================================

    section_header(
        "Production model",
        "The model currently used by the SuperLender inference pipeline.",
    )


    model_name = (
        MODEL_METADATA.get(
            "model_name"
        )
        or "HistGradientBoostingClassifier"
    )

    model_version = (
        MODEL_METADATA.get(
            "model_version"
        )
        or "v1"
    )

    decision_threshold = float(
        MODEL_METADATA.get(
            "decision_threshold",
            0.23,
        )
    )

    risk_event = (
        MODEL_METADATA.get(
            "risk_event"
        )
        or "Bad"
    )


    m1, m2, m3 = st.columns(3)


    with m1:

        metric_card(
            "Production model",
            "HistGradientBoosting",
            f"Classifier • Version {model_version}",
        )


    with m2:

        metric_card(
            "Risk event",
            risk_event,
            "Positive event for risk reporting",
        )


    with m3:

        metric_card(
            "Operating threshold",
            f"{decision_threshold:.1%}",
            "Configured production decision rule",
        )


    st.caption(
        f"Technical estimator: {model_name}"
    )


    # ========================================================
    # VALIDATION PERFORMANCE
    # ========================================================

    section_header(
        "Validation performance",
        (
            "Performance of the selected model on the held-out "
            f"validation period at the configured "
            f"{decision_threshold:.0%} threshold."
        )
    )


    validation_metrics = MODEL_EVIDENCE.get(
        "validation",
        {}
    )

    test_metrics = MODEL_EVIDENCE.get(
        "final_holdout",
        {}
    )


    v1, v2, v3, v4, v5 = st.columns(5)


    with v1:

        metric_card(
            "F1-score",
            format_metric(
                validation_metrics.get("f1"),
            ),
            "Bad class",
        )


    with v2:

        metric_card(
            "Bad recall",
            format_metric(
                validation_metrics.get("recall"),
                percentage=True,
            ),
            "Bad loans detected",
        )


    with v3:

        metric_card(
            "Bad precision",
            format_metric(
                validation_metrics.get("precision"),
                percentage=True,
            ),
            "Predicted Bad that were Bad",
        )


    with v4:

        metric_card(
            "ROC-AUC",
            format_metric(
                validation_metrics.get("roc_auc"),
            ),
            "Ranking performance",
        )


    with v5:

        metric_card(
            "PR-AUC",
            format_metric(
                validation_metrics.get("pr_auc"),
            ),
            "Bad-event ranking",
        )


    # ========================================================
    # VALIDATION VS TEST
    # ========================================================

    section_header(
        "Validation vs final holdout",
        "The final holdout was kept outside model and threshold "
        "selection and evaluated after the production configuration "
        "was frozen.",
    )


    comparison_data = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC",
                "PR-AUC",
            ],
            "Validation": [
                validation_metrics.get("accuracy"),
                validation_metrics.get("precision"),
                validation_metrics.get("recall"),
                validation_metrics.get("f1"),
                validation_metrics.get("roc_auc"),
                validation_metrics.get("pr_auc"),
            ],
            "Final holdout": [
                test_metrics.get("accuracy"),
                test_metrics.get("precision"),
                test_metrics.get("recall"),
                test_metrics.get("f1"),
                test_metrics.get("roc_auc"),
                test_metrics.get("pr_auc"),
            ],
        }
    )


    st.dataframe(
        comparison_data.style.format(
            {
                "Validation": "{:.3f}",
                "Final holdout": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


    st.caption(
        "Validation was used for model/threshold selection. "
        "The final holdout was evaluated after the configuration "
        "was frozen."
    )


    # ========================================================
    # CONFUSION MATRICES
    # ========================================================

    section_header(
        "Classification errors",
        "At the 23% operating threshold, the confusion matrix "
        "shows the trade-off between missed Bad loans and "
        "false Bad classifications.",
    )


    validation_cm = validation_metrics.get(
        "confusion_matrix"
    )

    test_cm = test_metrics.get(
        "confusion_matrix"
    )


    cm_left, cm_right = st.columns(2)


    with cm_left:

        st.markdown(
            "#### Validation"
        )

        if (
            isinstance(validation_cm, list)
            and len(validation_cm) == 2
        ):

            validation_cm_df = pd.DataFrame(
                validation_cm,
                index=[
                    "Actual Bad",
                    "Actual Good",
                ],
                columns=[
                    "Predicted Bad",
                    "Predicted Good",
                ],
            )

            st.dataframe(
                validation_cm_df,
                use_container_width=True,
            )

        else:

            st.info(
                "Validation confusion matrix is not available "
                "in the model metadata artifact."
            )


    with cm_right:

        st.markdown(
            "#### Final holdout"
        )

        if (
            isinstance(test_cm, list)
            and len(test_cm) == 2
        ):

            test_cm_df = pd.DataFrame(
                test_cm,
                index=[
                    "Actual Bad",
                    "Actual Good",
                ],
                columns=[
                    "Predicted Bad",
                    "Predicted Good",
                ],
            )

            st.dataframe(
                test_cm_df,
                use_container_width=True,
            )

        else:

            st.info(
                "Final-holdout confusion matrix is not available "
                "in the model metadata artifact."
            )


    # ========================================================
    # THRESHOLD ANALYSIS
    # ========================================================


    left, right = st.columns(
        [1.5, 1]
    )


    with left:

        st.subheader("Operating threshold")

        st.write(
            "The threshold converts the model's continuous Bad-risk score "
            "into a Good/Bad classification."
        )

        threshold_df = OPERATING_POINTS.copy()

        required_threshold_cols = {
            "Threshold",
            "Precision_Bad",
            "Recall_Bad",
            "F1_Bad",
        }

        if required_threshold_cols.issubset(threshold_df.columns):

            threshold_df = threshold_df.sort_values("Threshold").copy()

            # Keep the production model only.
            if "Model" in threshold_df.columns:
                threshold_df = threshold_df[
                    threshold_df["Model"]
                    .astype(str)
                    .str.contains("Tuned HistGradientBoosting", case=False, na=False)
                ].copy()

            threshold_df["Threshold"] = pd.to_numeric(
                threshold_df["Threshold"], errors="coerce"
            )

            for col in [
                "Precision_Bad",
                "Recall_Bad",
                "F1_Bad",
                "Flagged_as_Bad_%",
            ]:
                if col in threshold_df.columns:
                    threshold_df[col] = pd.to_numeric(
                        threshold_df[col], errors="coerce"
                    )

            threshold_df = threshold_df.dropna(subset=["Threshold"])

            production_threshold = float(
                MODEL_EVIDENCE["model"]["decision_threshold"]
            )

            # Find the actual production row in the artifact.
            production_rows = threshold_df[
                np.isclose(
                    threshold_df["Threshold"],
                    production_threshold,
                    atol=1e-8,
                )
            ]

            if not production_rows.empty:

                production_row = production_rows.iloc[0]

                metric_cols = [
                    "Precision_Bad",
                    "Recall_Bad",
                    "F1_Bad",
                ]

                chart_df = threshold_df[
                    ["Threshold"] + metric_cols
                ].copy()

                chart_df["Threshold (%)"] = (
                    chart_df["Threshold"] * 100
                )

                chart_df = chart_df.set_index("Threshold (%)")

                chart_df = chart_df.rename(
                    columns={
                        "Precision_Bad": "Precision",
                        "Recall_Bad": "Recall",
                        "F1_Bad": "F1",
                    }
                )

                st.line_chart(
                    chart_df[
                        ["Precision", "Recall", "F1"]
                    ],
                    use_container_width=True,
                )

                st.caption(
                    "As the threshold changes, precision, recall and F1 change "
                    "because the same continuous risk scores are converted into "
                    "different Good/Bad classifications."
                )

                # Production operating point
                st.markdown("**Production operating point**")

                op_col1, op_col2, op_col3, op_col4 = st.columns(4)

                op_col1.metric(
                    "Threshold",
                    f"{production_threshold:.1%}",
                )

                op_col2.metric(
                    "Bad precision",
                    f"{production_row['Precision_Bad']:.3f}",
                )

                op_col3.metric(
                    "Bad recall",
                    f"{production_row['Recall_Bad']:.3f}",
                )

                op_col4.metric(
                    "Bad F1",
                    f"{production_row['F1_Bad']:.3f}",
                )

                if "Flagged_as_Bad_%" in production_row.index:
                    st.caption(
                        f"Applications classified as Bad at this threshold: "
                        f"{production_row['Flagged_as_Bad_%']:.1f}%."
                    )

            else:
                st.warning(
                    "The production threshold could not be matched to the "
                    "threshold artifact."
                )

        else:
            st.warning(
                "The threshold artifact is missing one or more required "
                "columns for threshold analysis."
            )


    with right:

        st.subheader("Why 23%?")

        validation_evidence = MODEL_EVIDENCE["validation"]
        production_threshold = float(
            MODEL_EVIDENCE["model"]["decision_threshold"]
        )

        production_rows = OPERATING_POINTS[
            np.isclose(
                pd.to_numeric(
                    OPERATING_POINTS["Threshold"],
                    errors="coerce",
                ),
                production_threshold,
                atol=1e-8,
            )
        ].copy()

        st.write(
            "The model produces a continuous Bad-risk score. "
            "The operating threshold converts that score into the "
            "production Good/Bad classification."
        )

        st.write(
            f"At the configured {production_threshold:.0%} threshold, "
            f"scores below the threshold are classified as Good, while "
            f"scores at or above it are classified as Bad."
        )

        if not production_rows.empty:

            operating_row = production_rows.iloc[0]

            st.markdown(
                f"""
                The selected validation operating point produced:

                **Bad recall:** {operating_row["Recall_Bad"]:.1%}  
                **Bad precision:** {operating_row["Precision_Bad"]:.1%}  
                **Bad F1-score:** {operating_row["F1_Bad"]:.3f}  
                **Applications flagged as Bad:** {operating_row["Flagged_as_Bad_%"]:.1f}%
                """
            )

        st.write(
            "The 23% threshold was selected from validation analysis, "
            "where the threshold was treated as a decision parameter rather "
            "than being fixed at the default 50% probability cutoff."
        )

        st.write(
            "Changing the threshold changes the balance between false "
            "positive classifications and missed Bad applications. "
            "The final holdout was evaluated only after the model and "
            "operating threshold had been frozen."
        )


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    section_header(
        "What signals does the model use?",
        "Permutation importance measured how much validation "
        "PR-AUC changed when each feature was disrupted.",
    )


    if not PERMUTATION_IMPORTANCE.empty:

        importance_df = (
            PERMUTATION_IMPORTANCE.copy()
        )


        feature_column = next(
            (
                column
                for column in importance_df.columns
                if column.lower()
                in {
                    "feature",
                    "feature_name",
                }
            ),
            None,
        )

        importance_column = next(
            (
                column
                for column in importance_df.columns
                if column.lower()
                in {
                    "importance",
                    "mean_importance",
                    "permutation_importance",
                }
            ),
            None,
        )


        if (
            feature_column
            and importance_column
        ):

            importance_df[
                importance_column
            ] = pd.to_numeric(
                importance_df[
                    importance_column
                ],
                errors="coerce",
            )


            top_importance = (
                importance_df
                .dropna(
                    subset=[
                        importance_column
                    ]
                )
                .sort_values(
                    importance_column,
                    ascending=False,
                )
                .head(10)
                .sort_values(
                    importance_column,
                    ascending=True,
                )
            )


            fig, ax = plt.subplots(
                figsize=(8, 5)
            )


            ax.barh(
                top_importance[
                    feature_column
                ],
                top_importance[
                    importance_column
                ],
            )


            ax.set_xlabel(
                "Mean validation PR-AUC decrease"
            )

            ax.set_ylabel(
                "Feature"
            )

            ax.set_title(
                "Top validation permutation-importance signals"
            )


            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


            # Show a readable table too

            display_importance = (
                top_importance
                .sort_values(
                    importance_column,
                    ascending=False,
                )
                .copy()
            )


            display_importance = (
                display_importance[
                    [
                        feature_column,
                        importance_column,
                    ]
                ]
                .rename(
                    columns={
                        feature_column: "Feature",
                        importance_column: "Mean importance",
                    }
                )
            )


            st.dataframe(
                display_importance,
                use_container_width=True,
                hide_index=True,
            )


        else:

            st.info(
                "The permutation-importance artifact does not "
                "contain recognised feature/importance columns."
            )

    else:

        st.info(
            "The permutation-importance artifact is not available."
        )


    st.caption(
        "Permutation importance measures predictive usefulness "
        "for this model and validation sample. It does not prove "
        "that an individual feature causes default."
    )


    st.subheader("Subgroup monitoring")

    st.write(
        "Validation performance was also examined across selected demographic "
        "subgroups at the frozen 23% operating threshold."
    )

    st.caption(
        "These comparisons describe observed differences in model behaviour "
        "across groups. They do not establish that the model causes those "
        "differences or, by themselves, establish a fairness violation."
    )

    if not FAIRNESS_AUDIT.empty:

        dimensions = FAIRNESS_AUDIT["Dimension"].dropna().unique().tolist()

        selected_dimension = st.selectbox(
            "Audit dimension",
            dimensions,
            key="fairness_dimension",
        )

        fairness_view = FAIRNESS_AUDIT[
            FAIRNESS_AUDIT["Dimension"] == selected_dimension
        ].copy()

        display_fairness = fairness_view[
            [
                "Group",
                "N",
                "Bad_count",
                "Bad_rate",
                "Bad_recall",
                "False_positive_rate",
                "Bad_precision",
                "Predicted_Bad_rate",
            ]
        ].copy()

        display_fairness = display_fairness.rename(
            columns={
                "Bad_count": "Bad cases",
                "Bad_rate": "Actual Bad rate",
                "Bad_recall": "Bad recall",
                "False_positive_rate": "False-positive rate",
                "Bad_precision": "Bad precision",
                "Predicted_Bad_rate": "Predicted Bad rate",
            }
        )

        for col in [
            "Actual Bad rate",
            "Bad recall",
            "False-positive rate",
            "Bad precision",
            "Predicted Bad rate",
        ]:
            display_fairness[col] = (
                display_fairness[col] * 100
            ).round(1)

        st.dataframe(
            display_fairness,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Percentages are calculated on the validation subgroup. "
            "Subgroups with few Bad observations can produce unstable estimates "
            "and should be interpreted cautiously."
        )


    # ========================================================
    # CALIBRATION
    # ========================================================

    section_header(
        "Risk-score calibration",
        "The score is useful for ranking and classification, "
        "but should not be interpreted as a perfectly calibrated "
        "probability.",
    )


    calibration_left, calibration_right = st.columns(
        2
    )


    with calibration_left:

        st.markdown(
            "### Validation Brier score"
        )

        calibration_evidence = MODEL_EVIDENCE.get(
            "calibration",
            {}
        )

        brier_score = calibration_evidence.get(
            "brier_score"
        )

        if brier_score is not None:

            st.markdown(
                f"## {float(brier_score):.3f}"
            )

            st.caption(
                "Lower is better for probabilistic calibration."
            )

        else:

            st.info(
                "Calibration metric not available in metadata."
            )


    with calibration_right:

        st.write(
            calibration_evidence.get(
                "note",
                "Calibration evidence is not available"
            )
        )


    # ========================================================
    # LIMITATIONS
    # ========================================================

    section_header(
        "Important limitations",
        "Evidence that should remain visible alongside model output.",
    )


    l1, l2, l3, l4 = st.columns(4)


    with l1:

        st.markdown(
            "### Historical data coverage"
        )

        st.write(
            "Some current applications have no matching "
            "demographic records. Missing information is handled "
            "rather than filled from the test dataset."
        )


    with l2:

        st.markdown(
            "### Association is not causation"
        )

        st.write(
            "Feature importance and observed relationships "
            "describe model/data associations. They do not "
            "establish causal effects."
        )


    with l3:

        st.markdown(
            "### Decision support"
        )

        st.write(
            "The score is a model estimate intended to support "
            "credit-risk review. It is not a guaranteed lending "
            "decision."
        )

    
    with l4:

        st.markdown(
            "### Subgroup performance varies"
        )

        st.write(
            "Validation metrics were examined across selected demographic"
            "subgroups. Some groups show different recall and false-positive"
            "rates, but subgroup sample sizes and the observed validation data"
            "limit how strongly these differences can be interpreted."
        )