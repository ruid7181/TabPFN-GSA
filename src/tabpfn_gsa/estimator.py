from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted

from tabpfn_gsa.config import GSAConfig
from tabpfn_gsa.grid import (
    build_regular_grid_index,
    collect_neighbor_train_indices,
    merge_unique_indices,
    sample_non_neighbor_train_indices,
)


@dataclass(frozen=True)
class PredictionDiagnostics:
    """Summary diagnostics from one GSA prediction run."""

    average_train_size: float
    average_neighbor_size: float
    fitted_local_models: int
    fallback_predictions: int


@dataclass(frozen=True)
class PredictionResult:
    """Mean and uncertainty estimates from the GSA ensemble."""

    mean: np.ndarray
    std: np.ndarray
    diagnostics: PredictionDiagnostics


class GSARegressor(BaseEstimator, RegressorMixin):
    """A generic sklearn-style wrapper for the TabPFN-GSA sampling workflow."""

    def __init__(
        self,
        base_estimator: Any,
        spa_cols: list[str],
        x_cols: list[str] | None = None,
        K: int = 10,
        s: float = 0.1,
        n_ensembles: int = 8,
        min_random_samples: int = 2,
        include_spatial_features: bool = True,
        use_global_fallback: bool = True,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.base_estimator = base_estimator
        self.spa_cols = spa_cols
        self.x_cols = x_cols
        self.K = K
        self.s = s
        self.n_ensembles = n_ensembles
        self.min_random_samples = min_random_samples
        self.include_spatial_features = include_spatial_features
        self.use_global_fallback = use_global_fallback
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame | np.ndarray) -> "GSARegressor":
        """Store the training data and optionally fit a global fallback model."""

        X_df = self._validate_dataframe(X, variable_name="X")
        y_series = self._validate_target(y)
        self._validate_columns(X_df)

        x_cols = self.x_cols
        if x_cols is None:
            x_cols = [column for column in X_df.columns if column not in self.spa_cols]

        self.config_ = GSAConfig(
            K=self.K,
            s=self.s,
            n_ensembles=self.n_ensembles,
            min_random_samples=self.min_random_samples,
            include_spatial_features=self.include_spatial_features,
            use_global_fallback=self.use_global_fallback,
        )
        self.x_cols_ = list(x_cols)
        self.model_columns_ = self._build_model_columns()
        self.X_train_ = X_df.reset_index(drop=True).copy()
        self.y_train_ = y_series.reset_index(drop=True).copy()
        self.target_name_ = self.y_train_.name or "target"

        if self.config_.use_global_fallback:
            self.global_estimator_ = self._fit_estimator(self.X_train_, self.y_train_)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with the ensemble mean."""

        return self.predict_with_uncertainty(X).mean

    def predict_with_uncertainty(self, X: pd.DataFrame) -> PredictionResult:
        """Predict with ensemble mean, standard deviation, and diagnostics."""

        check_is_fitted(self, attributes=["X_train_", "y_train_", "x_cols_", "model_columns_"])
        X_df = self._validate_dataframe(X, variable_name="X")
        self._validate_columns(X_df)

        grid_index = build_regular_grid_index(
            train_coords=self.X_train_[self.spa_cols],
            test_coords=X_df[self.spa_cols],
            K=self.config_.K,
        )

        ensemble_predictions = np.full((self.config_.n_ensembles, len(X_df)), np.nan, dtype=float)
        train_sizes: list[int] = []
        neighbor_sizes: list[int] = []
        seed_sequence = np.random.SeedSequence(self.random_state)
        child_seeds = seed_sequence.spawn(self.config_.n_ensembles)

        for ensemble_index, child_seed in enumerate(child_seeds):
            rng = np.random.default_rng(child_seed)
            for grid_id, grid_cell in grid_index.items():
                if not grid_cell.test_indices:
                    continue

                neighbor_train_indices = collect_neighbor_train_indices(
                    grid_id=grid_id,
                    grid_index=grid_index,
                    K=self.config_.K,
                )
                n_random_samples = self._compute_random_sample_count(len(neighbor_train_indices))
                random_train_indices = sample_non_neighbor_train_indices(
                    grid_id=grid_id,
                    grid_index=grid_index,
                    K=self.config_.K,
                    n_samples=n_random_samples,
                    rng=rng,
                )
                local_train_indices = merge_unique_indices(neighbor_train_indices, random_train_indices)

                if not local_train_indices:
                    if self.config_.use_global_fallback:
                        predictions = self.global_estimator_.predict(X_df.iloc[grid_cell.test_indices][self.model_columns_])
                        ensemble_predictions[ensemble_index, grid_cell.test_indices] = np.asarray(predictions).reshape(-1)
                        continue
                    raise RuntimeError(
                        "No local training samples were available for a test cell and global fallback is disabled."
                    )

                local_estimator = self._fit_estimator(
                    self.X_train_.iloc[local_train_indices],
                    self.y_train_.iloc[local_train_indices],
                )
                predictions = local_estimator.predict(X_df.iloc[grid_cell.test_indices][self.model_columns_])
                ensemble_predictions[ensemble_index, grid_cell.test_indices] = np.asarray(predictions).reshape(-1)
                train_sizes.append(len(local_train_indices))
                neighbor_sizes.append(len(neighbor_train_indices))

        fallback_predictions = 0
        if np.isnan(ensemble_predictions).any():
            if not self.config_.use_global_fallback:
                raise RuntimeError("Missing predictions were produced and global fallback is disabled.")

            missing_mask = np.isnan(ensemble_predictions)
            row_has_missing = missing_mask.any(axis=0)
            fallback_predictions = int(row_has_missing.sum())
            fallback_values = np.asarray(self.global_estimator_.predict(X_df.loc[row_has_missing, self.model_columns_]))
            ensemble_predictions[:, row_has_missing] = fallback_values

        diagnostics = PredictionDiagnostics(
            average_train_size=float(np.mean(train_sizes)) if train_sizes else 0.0,
            average_neighbor_size=float(np.mean(neighbor_sizes)) if neighbor_sizes else 0.0,
            fitted_local_models=len(train_sizes),
            fallback_predictions=fallback_predictions,
        )

        return PredictionResult(
            mean=ensemble_predictions.mean(axis=0),
            std=ensemble_predictions.std(axis=0),
            diagnostics=diagnostics,
        )

    def _fit_estimator(self, X: pd.DataFrame, y: pd.Series) -> Any:
        estimator = clone(self.base_estimator)
        estimator.fit(X[self.model_columns_], y)
        return estimator

    def _build_model_columns(self) -> list[str]:
        if self.config_.include_spatial_features:
            return [*self.x_cols_, *self.spa_cols]
        return list(self.x_cols_)

    def _compute_random_sample_count(self, n_neighbor_samples: int) -> int:
        if self.config_.s == 0.0:
            return self.config_.min_random_samples
        remaining_train_pool = max(0, len(self.X_train_) - n_neighbor_samples)
        return int(remaining_train_pool * self.config_.s)

    def _validate_columns(self, X: pd.DataFrame) -> None:
        missing_columns = [column for column in self.spa_cols if column not in X.columns]
        if missing_columns:
            raise ValueError(f"Missing required spatial columns: {missing_columns}")

        if self.x_cols is not None:
            feature_missing = [column for column in self.x_cols if column not in X.columns]
            if feature_missing:
                raise ValueError(f"Missing required x columns: {feature_missing}")

        if len(self.spa_cols) != 2:
            raise ValueError("Exactly two spa columns are required.")

    @staticmethod
    def _validate_dataframe(X: Any, variable_name: str) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{variable_name} must be a pandas DataFrame.")
        return X.copy()

    @staticmethod
    def _validate_target(y: pd.Series | pd.DataFrame | np.ndarray) -> pd.Series:
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("Only a single regression target is supported.")
            return y.iloc[:, 0].copy()
        if isinstance(y, pd.Series):
            return y.copy()

        y_array = np.asarray(y).reshape(-1)
        return pd.Series(y_array, name="target")
