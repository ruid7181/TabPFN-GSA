from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Sequence

import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold


@dataclass(frozen=True)
class GSATuningResult:
    """Result of Optuna-based tuning for the two main GSA hyperparameters."""

    best_estimator: Any
    best_params: dict[str, Any]
    best_score: float
    trials_dataframe: pd.DataFrame
    metric: str
    study: optuna.Study


def tune_gsa(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    K_values: Sequence[int] | None = None,
    s_values: Sequence[float] | None = None,
    metric: str = "mae",
    cv: int = 3,
    n_trials: int = 20,
    random_state: int | None = 0,
    refit: bool = True,
    verbose: bool = False,
) -> GSATuningResult:
    """Tune only the two main GSA hyperparameters with Optuna.

    Parameters are intentionally simple:
    - `K_values`: candidate values for the number of grid cells per axis
    - `s_values`: candidate values for the distant grid sampling rate
    - `metric`: one of `mae`, `mse`, `rmse`, or `r2`
    """

    X_df, y_series = _prepare_supervised_inputs(X, y)
    scoring_name = _resolve_metric(metric)
    scorer = get_scorer(scoring_name)
    splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    K_values = list(K_values or [4, 6, 8, 10, 12])
    s_values = list(s_values or [0.0, 0.05, 0.1, 0.2, 0.3])

    if not K_values:
        raise ValueError("K_values must contain at least one candidate value.")
    if not s_values:
        raise ValueError("s_values must contain at least one candidate value.")

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "K": trial.suggest_categorical("K", K_values),
            "s": trial.suggest_categorical("s", s_values),
        }

        fold_scores: list[float] = []
        if verbose:
            print(f"Trial {trial.number}: {params}")

        for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(X_df, y_series)):
            model = clone(estimator)
            model.set_params(**params)

            X_train = X_df.iloc[train_idx].reset_index(drop=True)
            y_train = y_series.iloc[train_idx].reset_index(drop=True)
            X_valid = X_df.iloc[valid_idx].reset_index(drop=True)
            y_valid = y_series.iloc[valid_idx].reset_index(drop=True)

            model.fit(X_train, y_train)
            fold_score = float(scorer(model, X_valid, y_valid))
            fold_scores.append(fold_score)

            if verbose:
                print(f"  fold={fold_index + 1}/{cv} score={fold_score:.6f}")

        mean_score = sum(fold_scores) / len(fold_scores)
        trial.set_user_attr("fold_scores", fold_scores)
        return mean_score

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_estimator = clone(estimator)
    best_estimator.set_params(**study.best_params)
    if refit:
        best_estimator.fit(X_df, y_series)

    return GSATuningResult(
        best_estimator=best_estimator,
        best_params=dict(study.best_params),
        best_score=float(study.best_value),
        trials_dataframe=study.trials_dataframe(),
        metric=metric.lower(),
        study=study,
    )


def _prepare_supervised_inputs(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("Only a single regression target is supported.")
        y_series = y.iloc[:, 0].reset_index(drop=True)
    else:
        y_series = pd.Series(y).reset_index(drop=True)

    return X.reset_index(drop=True).copy(), y_series


def _resolve_metric(metric: str) -> str:
    metric_aliases = {
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    normalized = metric.lower()
    if normalized not in metric_aliases:
        raise ValueError("metric must be one of: 'mae', 'mse', 'rmse', or 'r2'.")
    return metric_aliases[normalized]
