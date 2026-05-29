from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabpfn_gsa import GSAModel


def make_dataset(n_samples: int = 80) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n_samples),
            "x2": rng.normal(size=n_samples),
            "coord_x": rng.uniform(0.0, 1.0, size=n_samples),
            "coord_y": rng.uniform(0.0, 1.0, size=n_samples),
        }
    )
    y = pd.Series(2.0 * X["x1"] - X["x2"], name="target")
    return X.iloc[:50].reset_index(drop=True), y.iloc[:50].reset_index(drop=True), X.iloc[50:].reset_index(drop=True)


class MeanFunctionModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.value)


def fit_mean_model(X, y, offset: float = 0.0) -> MeanFunctionModel:
    return MeanFunctionModel(float(np.mean(y)) + offset)


def predict_mean_model(model: MeanFunctionModel, X, scale: float = 1.0) -> np.ndarray:
    return model.predict(X) * scale


def test_unified_model_accepts_custom_fit_and_predict_functions() -> None:
    X_train, y_train, X_test = make_dataset()
    model = GSAModel(
        spa_cols=["coord_x", "coord_y"],
        x_cols=["x1", "x2"],
        K=4,
        s=0.1,
        random_state=0,
        model_kwargs={
            "fit_fn": fit_mean_model,
            "predict_fn": predict_mean_model,
            "fit_kwargs": {"offset": 1.0},
            "predict_kwargs": {"scale": 0.5},
        },
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert predictions.shape == (len(X_test),)
    assert np.isfinite(predictions).all()
    assert model.get_runtime_info().estimator_class == "fit_fn/predict_fn"
    assert model.get_runtime_info().backend_name == "custom"


def test_custom_function_model_requires_fit_and_predict_functions() -> None:
    with pytest.raises(ValueError, match="both fit_fn and predict_fn"):
        GSAModel(
            spa_cols=["coord_x", "coord_y"],
            x_cols=["x1", "x2"],
            model_kwargs={"fit_fn": fit_mean_model},
        )


def test_custom_model_rejects_estimator_argument() -> None:
    with pytest.raises(ValueError, match="Do not pass estimator"):
        GSAModel(
            spa_cols=["coord_x", "coord_y"],
            x_cols=["x1", "x2"],
            model_kwargs={
                "estimator": object(),
                "fit_fn": fit_mean_model,
                "predict_fn": predict_mean_model,
            },
        )


def test_custom_model_rejects_unused_kwargs() -> None:
    with pytest.raises(ValueError, match="Unexpected custom model_kwargs keys"):
        GSAModel(
            spa_cols=["coord_x", "coord_y"],
            x_cols=["x1", "x2"],
            model_kwargs={
                "fit_fn": fit_mean_model,
                "predict_fn": predict_mean_model,
                "unused": True,
            },
        )


def test_custom_function_fit_must_return_state() -> None:
    def bad_fit(X, y):
        return None

    def predict_fn(model, X):
        return np.zeros(len(X))

    model = GSAModel(
        spa_cols=["coord_x", "coord_y"],
        x_cols=["x1", "x2"],
        model_kwargs={"fit_fn": bad_fit, "predict_fn": predict_fn},
    )
    X_train, y_train, _ = make_dataset()

    with pytest.raises(ValueError, match="fit_fn must return"):
        model.fit(X_train, y_train)
