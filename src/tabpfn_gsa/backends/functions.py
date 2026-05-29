from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from tabpfn_gsa.backends.base import BuiltModelBackend, ModelRuntimeInfo


class _FunctionRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-style wrapper around user-provided fit and predict functions."""

    def __init__(
        self,
        fit_fn: Callable[..., Any],
        predict_fn: Callable[..., Any],
        fit_kwargs: dict[str, Any] | None = None,
        predict_kwargs: dict[str, Any] | None = None,
        sends_data_to_remote: bool = False,
    ) -> None:
        self.fit_fn = fit_fn
        self.predict_fn = predict_fn
        self.fit_kwargs = fit_kwargs
        self.predict_kwargs = predict_kwargs
        self.sends_data_to_remote = sends_data_to_remote

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "_FunctionRegressor":
        fitted_model = self.fit_fn(X, y, **dict(self.fit_kwargs or {}))
        if fitted_model is None:
            raise ValueError("custom fit_fn must return a fitted model or fitted state.")
        self.fitted_model_ = fitted_model
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not hasattr(self, "fitted_model_"):
            raise ValueError("This custom model has not been fitted yet.")
        predictions = self.predict_fn(self.fitted_model_, X, **dict(self.predict_kwargs or {}))
        return np.asarray(predictions).reshape(-1)


def build_function_backend(model_kwargs: dict[str, Any]) -> BuiltModelBackend:
    """Build a generic model backend from user-provided fit/predict functions."""

    kwargs = dict(model_kwargs)
    fit_fn = kwargs.pop("fit_fn", None)
    predict_fn = kwargs.pop("predict_fn", None)
    fit_kwargs = kwargs.pop("fit_kwargs", None)
    predict_kwargs = kwargs.pop("predict_kwargs", None)
    sends_data_to_remote = bool(kwargs.pop("sends_data_to_remote", False))

    if "estimator" in kwargs:
        raise ValueError("Use fit_fn and predict_fn for custom models. Do not pass estimator.")

    if fit_fn is None or predict_fn is None:
        raise ValueError("Custom models require both fit_fn and predict_fn.")

    if kwargs:
        unknown_keys = ", ".join(sorted(kwargs))
        raise ValueError(
            "Unexpected custom model_kwargs keys: "
            f"{unknown_keys}. Use only fit_fn, predict_fn, fit_kwargs, predict_kwargs, and sends_data_to_remote."
        )

    estimator = _FunctionRegressor(
        fit_fn=fit_fn,
        predict_fn=predict_fn,
        fit_kwargs=fit_kwargs,
        predict_kwargs=predict_kwargs,
        sends_data_to_remote=sends_data_to_remote,
    )

    runtime_info = ModelRuntimeInfo(
        backend_name="custom",
        requested_execution="user-provided",
        resolved_execution="user-provided",
        estimator_class="fit_fn/predict_fn",
        package_name="user-provided",
        package_version=None,
        model_version=None,
        torch_package_version=None,
        requested_device=None,
        available_local_devices=(),
        sends_data_to_remote=sends_data_to_remote,
        phe_enabled=False,
        model_kwargs=_summarize_custom_kwargs(model_kwargs),
        notes=(
            "User-provided fit_fn/predict_fn model backend.",
            "fit_fn must return fitted state; predict_fn receives that fitted state and X.",
        ),
    )
    return BuiltModelBackend(estimator=estimator, runtime_info=runtime_info)


def _summarize_custom_kwargs(model_kwargs: dict[str, Any]) -> dict[str, str]:
    summary: dict[str, str] = {}
    for key, value in model_kwargs.items():
        if callable(value):
            summary[key] = getattr(value, "__name__", type(value).__name__)
        else:
            summary[key] = type(value).__name__
    return summary
