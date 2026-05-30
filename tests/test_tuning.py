from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from tabpfn_gsa import tune_gsa
from tabpfn_gsa.estimator import GSARegressor


def make_dataset(n_samples: int = 180) -> tuple[pd.DataFrame, pd.Series]:
    import numpy as np

    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n_samples),
            "x2": rng.normal(size=n_samples),
            "coord_x": rng.uniform(0.0, 1.0, size=n_samples),
            "coord_y": rng.uniform(0.0, 1.0, size=n_samples),
        }
    )
    y = 1.2 * X["x1"] - 0.4 * X["x2"] + X["coord_x"] * 0.5 - X["coord_y"] * 0.3
    return X, y


def test_tune_gsa_searches_grid_and_refits() -> None:
    X, y = make_dataset()
    estimator = GSARegressor(
        base_estimator=RandomForestRegressor(n_estimators=10, random_state=0),
        spa_cols=["coord_x", "coord_y"],
        x_cols=["x1", "x2"],
        random_state=0,
    )

    result = tune_gsa(
        estimator=estimator,
        X=X,
        y=y,
        K_values=[4, 9],
        s_values=[0.0, 0.1],
        cv=2,
        n_trials=4,
    )

    assert set(result.best_params.keys()) == {"K", "s"}
    assert not result.trials_dataframe.empty
    assert result.best_estimator.K == result.best_params["K"]
    assert result.best_estimator.s == result.best_params["s"]
    assert "params_K" in result.trials_dataframe.columns


def test_tune_gsa_supports_metric_aliases() -> None:
    X, y = make_dataset()
    estimator = GSARegressor(
        base_estimator=RandomForestRegressor(n_estimators=10, random_state=0),
        spa_cols=["coord_x", "coord_y"],
        x_cols=["x1", "x2"],
        random_state=0,
    )

    result = tune_gsa(
        estimator=estimator,
        X=X,
        y=y,
        K_values=[4],
        s_values=[0.1],
        metric="r2",
        cv=2,
        n_trials=1,
    )

    assert result.metric == "r2"
    assert len(result.trials_dataframe) == 1


def test_tune_gsa_accepts_mse_metric() -> None:
    X, y = make_dataset()
    estimator = GSARegressor(
        base_estimator=RandomForestRegressor(n_estimators=10, random_state=0),
        spa_cols=["coord_x", "coord_y"],
        x_cols=["x1", "x2"],
        random_state=0,
    )

    result = tune_gsa(
        estimator=estimator,
        X=X,
        y=y,
        K_values=[4, 9],
        s_values=[0.0, 0.1],
        metric="mse",
        cv=2,
        n_trials=3,
        random_state=0,
    )

    assert "K" in result.best_params
    assert "s" in result.best_params
    assert not result.trials_dataframe.empty
