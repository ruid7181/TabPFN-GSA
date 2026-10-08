from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

from tabpfn_gsa.backends.base import BuiltModelBackend, ModelRuntimeInfo


class _FunctionRegressor(RegressorMixin, BaseEstimator):
    """Sklearn-style wrapper around user-provided fit and predict functions."""

    def __init__(
        self,
        fit_fn: Callable[..., Any],
        predict_fn: Callable[..., Any] | None = None,
        fit_kwargs: dict[str, Any] | None = None,
        predict_kwargs: dict[str, Any] | None = None,
        sends_data_to_remote: bool = False,
    ) -> None:
        self.fit_fn = fit_fn
        self.predict_fn = predict_fn
        self.fit_kwargs = fit_kwargs
        self.predict_kwargs = predict_kwargs
        self.sends_data_to_remote = sends_data_to_remote

    def __sklearn_clone__(self) -> "_FunctionRegressor":
        # Keep callbacks together without copying loaded models or GPU state.
        return type(self)(
            fit_fn=self.fit_fn,
            predict_fn=self.predict_fn,
            fit_kwargs=deepcopy(self.fit_kwargs),
            predict_kwargs=deepcopy(self.predict_kwargs),
            sends_data_to_remote=self.sends_data_to_remote,
        )

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "_FunctionRegressor":
        fit_fn = self.fit_fn
        source = getattr(fit_fn, "__self__", None)
        if source is not None and not isinstance(source, type):
            try:
                local_model = clone(source)
            except Exception as error:
                raise TypeError(
                    "This model cannot be cloned for independent grid contexts. "
                    "Use a fit_fn that creates and returns a fresh model or context state."
                ) from error
            fit_fn = getattr(local_model, fit_fn.__name__)
        fitted_model = fit_fn(X, y, **dict(self.fit_kwargs or {}))
        if fitted_model is None and source is not None and not isinstance(source, type):
            fitted_model = local_model
        if fitted_model is None:
            raise ValueError("custom fit_fn must return a fitted model or fitted state.")
        self.fitted_model_ = fitted_model
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not hasattr(self, "fitted_model_"):
            raise ValueError("This custom model has not been fitted yet.")
        predict_fn = self.predict_fn
        if predict_fn is None:
            predict_fn = getattr(self.fitted_model_, "predict", None)
            if not callable(predict_fn):
                raise ValueError(
                    "The state returned by fit_fn has no predict method. "
                    "Provide predict_fn(state, X_test) in model_kwargs."
                )
            return predict_fn(X, **dict(self.predict_kwargs or {}))
        source = getattr(self.fit_fn, "__self__", None)
        if source is not None and getattr(predict_fn, "__self__", None) is source:
            return getattr(self.fitted_model_, predict_fn.__name__)(
                X, **dict(self.predict_kwargs or {})
            )
        return predict_fn(self.fitted_model_, X, **dict(self.predict_kwargs or {}))


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

    if not callable(fit_fn):
        raise ValueError("Custom models require a callable fit_fn in model_kwargs.")
    if predict_fn is not None and not callable(predict_fn):
        raise ValueError("predict_fn must be callable when provided.")
    for key, value in (("fit_kwargs", fit_kwargs), ("predict_kwargs", predict_kwargs)):
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"{key} must be a dictionary when provided.")
    fit_source = getattr(fit_fn, "__self__", None)
    predict_source = getattr(predict_fn, "__self__", None)
    if fit_source is not None and predict_source is not None and fit_source is not predict_source:
        raise ValueError("fit_fn and predict_fn must be methods of the same model instance.")

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
            "Pass a model's bound methods or functions that create and query context state.",
            "predict_fn is optional when the fitted model has a predict method.",
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
