from __future__ import annotations

import numpy as np
import pandas as pd
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
        K=6,
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
        K=5,
        s=0.0,
        n_ensembles=2,
        random_state=0,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert model.x_cols_ == ["x1", "x2"]
    assert predictions.shape == (len(X_test),)
    assert np.isfinite(predictions).all()
