from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CurrentLoanInput(BaseModel):
    """Minimum current-loan fields required by FeatureEngineer.

    Extra fields are accepted so the API can receive a full current-loan
    record from the source dataset without losing columns that are not used
    by the production feature contract.
    """

    model_config = ConfigDict(extra="allow")

    systemloanid: int | str
    customerid: int | str
    creationdate: str
    loanamount: float = Field(gt=0)
    totaldue: float = Field(gt=0)
    termdays: float = Field(gt=0)
    loannumber: float = Field(gt=0)
    referredby: str | None = None

    @model_validator(mode="after")
    def validate_loan_amount_relationship(self) -> "CurrentLoanInput":
        if self.loanamount > self.totaldue:
            raise ValueError("loanamount must not exceed totaldue")
        return self

    @field_validator("creationdate")
    @classmethod
    def validate_creationdate(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("creationdate must not be empty")
        return value


class SinglePredictionRequest(BaseModel):
    """Request body for one current-loan risk prediction."""

    current_loan: CurrentLoanInput
    previous_loans: list[dict[str, Any]] = Field(default_factory=list)
    demographics: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def reject_target_and_duplicate_application(self) -> "SinglePredictionRequest":
        current = self.current_loan.model_dump()
        if "good_bad_flag" in current:
            raise ValueError(
                "good_bad_flag must not be supplied to the prediction workflow"
            )
        return self


class BatchPredictionRequest(BaseModel):
    """Request body for batch current-loan risk prediction."""

    current_loans: list[CurrentLoanInput] = Field(min_length=1)
    previous_loans: list[dict[str, Any]] = Field(default_factory=list)
    demographics: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_batch_contract(self) -> "BatchPredictionRequest":
        loan_rows = [row.model_dump() for row in self.current_loans]

        for row_number, row in enumerate(loan_rows, start=1):
            if "good_bad_flag" in row:
                raise ValueError(
                    "good_bad_flag must not be supplied to the prediction workflow "
                    f"(row {row_number})"
                )

        ids = [row["systemloanid"] for row in loan_rows]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "systemloanid must be unique within the uploaded batch"
            )

        return self


class PredictionResponse(BaseModel):
    """Single application prediction returned by the API."""

    systemloanid: int | str
    customerid: int | str
    predicted_class: Literal["Good", "Bad"]
    bad_risk_score: float = Field(ge=0, le=1)
    decision_threshold: float = Field(ge=0, le=1)
    model_name: str
    model_version: str
    risk_event: Literal["Bad"] = "Bad"


class BatchPredictionResponse(BaseModel):
    """Batch prediction response with one result per current application."""

    count: int = Field(ge=0)
    predictions: list[PredictionResponse]
