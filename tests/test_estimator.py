from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from tabpfn_gsa.estimator import GSARegressor


def make_dataset(n_samples: int = 240, random_state: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    coord_x = rng.uniform(0.0, 1.0, size=n_samples)
    coord_y = rng.uniform(0.0, 1.0, size=n_samples)
    x1 = rng.normal(size=n_samples)
    x2 = rng.normal(size=n_samples)
    target = 2.0 * x1 - 0.5 * x2 + np.sin(coord_x * 4.0) + np.cos(coord_y * 3.0)

    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "coord_x": coord_x,
            "coord_y": coord_y,
        }
    )
    y = pd.Series(target, name="target")

    split = int(n_samples * 0.7)
    return X.iloc[:split].reset_index(drop=True), y.iloc[:split].reset_index(drop=True), X.iloc[split:].reset_index(drop=True)


def test_gsa_regressor_returns_prediction_statistics() -> None:
    X_train, y_train, X_test = make_dataset()
    model = GSARegressor(
        base_estimator=RandomForestRegressor(n_estimators=20, random_state=0),
        spa_cols=["coord_x", "coord_y"],
        x_cols=["x1", "x2"],
        K=9,
        s=0.1,
        n_ensembles=3,
        random_state=0,
    )

    model.fit(X_train, y_train)
    result = model.predict_with_uncertainty(X_test)

    assert result.mean.shape == (len(X_test),)
    assert result.std.shape == (len(X_test),)
    assert np.isfinite(result.mean).all()
    assert np.all(result.std >= 0.0)
    assert result.diagnostics.fitted_local_models > 0


def test_gsa_regressor_infers_x_cols_from_dataframe() -> None:
    X_train, y_train, X_test = make_dataset()
    model = GSARegressor(
        base_estimator=RandomForestRegressor(n_estimators=10, random_state=0),
        spa_cols=["coord_x", "coord_y"],
        x_cols=None,
        K=4,
        s=0.0,
        n_ensembles=2,
        random_state=0,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert model.x_cols_ == ["x1", "x2"]
    assert predictions.shape == (len(X_test),)
    assert np.isfinite(predictions).all()


def test_gsa_regressor_requires_square_K() -> None:
    X_train, y_train, _ = make_dataset()
    model = GSARegressor(
        base_estimator=RandomForestRegressor(n_estimators=10, random_state=0),
        spa_cols=["coord_x", "coord_y"],
        x_cols=["x1", "x2"],
        K=6,
        random_state=0,
    )

    with pytest.raises(ValueError, match="K must be a square number"):
        model.fit(X_train, y_train)


def test_prediction_is_stable_when_batch_contains_out_of_bounds_points() -> None:
    X_train, y_train, _ = make_dataset(n_samples=400)
    in_bounds_point = pd.DataFrame(
        {
            "x1": [0.0],
            "x2": [0.0],
            "coord_x": [0.9],
            "coord_y": [0.9],
        }
    )
    far_point = pd.DataFrame(
        {
            "x1": [0.0],
            "x2": [0.0],
            "coord_x": [100.0],
            "coord_y": [100.0],
        }
    )
    batch = pd.concat([in_bounds_point, far_point], ignore_index=True)
    model = GSARegressor(
        base_estimator=RandomForestRegressor(n_estimators=10, random_state=0),
        spa_cols=["coord_x", "coord_y"],
        x_cols=["x1", "x2"],
        K=16,
        s=0.0,
        n_ensembles=1,
        random_state=0,
    )

    model.fit(X_train, y_train)
    single_prediction = model.predict(in_bounds_point)
    batch_prediction = model.predict(batch)

    assert np.allclose(single_prediction[0], batch_prediction[0])


def test_gsa_regressor_accepts_explicit_spatial_bounds() -> None:
    X_train, y_train, _ = make_dataset()
    model = GSARegressor(
        base_estimator=RandomForestRegressor(n_estimators=10, random_state=0),
        spa_cols=["coord_x", "coord_y"],
        spa_bounds={"coord_x": (-1.0, 2.0), "coord_y": (-2.0, 3.0)},
        x_cols=["x1", "x2"],
        K=4,
        random_state=0,
    )

    model.fit(X_train, y_train)

    assert np.allclose(model.spa_bounds_.mins, [-1.0, -2.0])
    assert np.allclose(model.spa_bounds_.maxs, [2.0, 3.0])
