from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np


class ModelManager:
    """Load and expose the frozen SuperLender production model artifact."""

    DEFAULT_THRESHOLD = 0.23
    DEFAULT_VERSION = "v1"
    DEFAULT_RISK_EVENT = "Bad"

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = (
            Path(metadata_path)
            if metadata_path is not None
            else None
        )
        self._model: Any | None = None
        self._metadata: dict[str, Any] | None = None

    @property
    def model(self) -> Any:
        """Return the fitted model, loading it once when first needed."""
        if self._model is None:
            self._load()
        return self._model

    @property
    def metadata(self) -> dict[str, Any]:
        """Return model metadata, defaulting to an empty mapping when absent."""
        if self._metadata is None:
            self._load_metadata()
        return self._metadata

    def _load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {self.model_path}"
            )

        try:
            self._model = joblib.load(self.model_path)
        except Exception as exc:  # pragma: no cover - defensive error boundary
            raise RuntimeError(
                f"Could not load model artifact: {self.model_path}"
            ) from exc

        if not hasattr(self._model, "predict_proba"):
            raise TypeError(
                "Loaded model artifact must expose predict_proba()."
            )

        self._load_metadata()

    def _load_metadata(self) -> None:
        if self._metadata is not None:
            return

        if self.metadata_path is None or not self.metadata_path.exists():
            self._metadata = {}
            return

        try:
            with self.metadata_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Could not read model metadata: {self.metadata_path}"
            ) from exc

        if not isinstance(data, dict):
            raise TypeError("Model metadata must contain a JSON object.")

        self._metadata = data

    def _final_estimator(self) -> Any:
        model = self.model
        named_steps = getattr(model, "named_steps", None)
        if isinstance(named_steps, dict) and "model" in named_steps:
            return named_steps["model"]
        return model

    def classes(self) -> list[Any]:
        estimator = self._final_estimator()
        classes = getattr(estimator, "classes_", None)
        if classes is None:
            classes = getattr(self.model, "classes_", None)
        if classes is None:
            raise RuntimeError("The loaded model does not expose classes_.")
        return list(classes)

    def bad_class_index(self) -> int:
        classes = self.classes()
        if 0 not in classes:
            raise RuntimeError(
                "The loaded production model does not contain expected Bad class 0."
            )
        return classes.index(0)

    def predict_proba(self, X) -> np.ndarray:
        probabilities = np.asarray(self.model.predict_proba(X), dtype=float)
        if probabilities.ndim != 2:
            raise RuntimeError("Model predict_proba() returned an invalid shape.")
        return probabilities

    def model_name(self) -> str:
        name = (
            self.metadata.get("model_name")
            or self.metadata.get("name")
        )
        if name:
            return str(name)
        return self._final_estimator().__class__.__name__

    def model_version(self) -> str:
        version = (
            self.metadata.get("model_version")
            or self.metadata.get("version")
            or self.DEFAULT_VERSION
        )
        return str(version)

    def decision_threshold(self) -> float:
        value = self.metadata.get(
            "decision_threshold",
            self.DEFAULT_THRESHOLD,
        )
        try:
            threshold = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Model metadata contains an invalid decision_threshold."
            ) from exc

        if not 0 <= threshold <= 1:
            raise RuntimeError(
                "decision_threshold must be between 0 and 1."
            )
        return threshold

    def risk_event(self) -> str:
        return str(
            self.metadata.get(
                "risk_event",
                self.DEFAULT_RISK_EVENT,
            )
        )

    def info(self) -> dict[str, Any]:
        """Return deployment metadata without exposing model internals."""
        return {
            "model_name": self.model_name(),
            "model_version": self.model_version(),
            "risk_event": self.risk_event(),
            "decision_threshold": self.decision_threshold(),
            "artifact": self.model_path.name,
            "metadata_artifact": (
                self.metadata_path.name
                if self.metadata_path is not None
                else None
            ),
        }
