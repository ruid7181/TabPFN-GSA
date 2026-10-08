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
    build_grid_bounds,
    collect_neighbor_train_indices,
    GridBounds,
    merge_unique_indices,
    sample_non_neighbor_train_indices,
)


class _EmptyContextError(ValueError):
    """A grid has no training context for the selected K and s."""


@dataclass(frozen=True)
class PredictionDiagnostics:
    """Summary diagnostics from one GSA prediction run."""

    average_train_size: float
    average_neighbor_size: float
    fitted_local_models: int


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
        spa_bounds: dict[str, tuple[float, float]] | None = None,
        x_cols: list[str] | None = None,
        K: int = 64,
        s: float = 0.1,
        n_ensembles: int = 8,
        min_random_samples: int = 2,
        include_spatial_features: bool = True,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.base_estimator = base_estimator
        self.spa_cols = spa_cols
        self.spa_bounds = spa_bounds
        self.x_cols = x_cols
        self.K = K
        self.s = s
        self.n_ensembles = n_ensembles
        self.min_random_samples = min_random_samples
        self.include_spatial_features = include_spatial_features
        self.random_state = random_state
        self.verbose = verbose

    def fit(
        self, X: pd.DataFrame, y: pd.Series | pd.DataFrame | np.ndarray
    ) -> "GSARegressor":
        """Validate and store training data; local models are fitted at prediction time."""

        X_df = self._validate_dataframe(X, variable_name="X")
        y_series = self._validate_target(y)
        if len(X_df) != len(y_series):
            raise ValueError(
                f"X and y have inconsistent numbers of samples: {len(X_df)} != {len(y_series)}."
            )
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
        )
        self.x_cols_ = list(x_cols)
        self.model_columns_ = self._build_model_columns()
        self.X_train_ = X_df.reset_index(drop=True).copy()
        self.y_train_ = y_series.reset_index(drop=True).copy()
        self.target_name_ = self.y_train_.name or "target"
        self.spa_bounds_ = self._resolve_spa_bounds(self.X_train_[self.spa_cols])

        self._sampling_seed_ = np.random.SeedSequence(self.random_state).entropy

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with the ensemble mean."""

        return self.predict_with_uncertainty(X).mean

    def predict_with_uncertainty(self, X: pd.DataFrame) -> PredictionResult:
        """Predict with ensemble mean, standard deviation, and diagnostics."""

        check_is_fitted(
            self,
            attributes=[
                "X_train_",
                "y_train_",
                "x_cols_",
                "model_columns_",
                "spa_bounds_",
            ],
        )
        X_df = self._validate_dataframe(X, variable_name="X")
        self._validate_columns(X_df)
        missing_columns = [col for col in self.model_columns_ if col not in X_df]
        if missing_columns:
            raise ValueError(f"Missing training feature columns: {missing_columns}")

        grid_index = build_regular_grid_index(
            train_coords=self.X_train_[self.spa_cols],
            test_coords=X_df[self.spa_cols],
            K=self.config_.n_grid_per_axis,
            bounds=self.spa_bounds_,
        )

        ensemble_predictions = np.full(
            (self.config_.n_ensembles, len(X_df)), np.nan, dtype=float
        )
        train_sizes: list[int] = []
        neighbor_sizes: list[int] = []
        for ensemble_index in range(self.config_.n_ensembles):
            for grid_id, grid_cell in grid_index.items():
                if not grid_cell.test_indices:
                    continue

                # Each cell gets the same context regardless of prediction batching.
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        self._sampling_seed_, spawn_key=(ensemble_index, *grid_id)
                    )
                )
                neighbor_train_indices = collect_neighbor_train_indices(
                    grid_id=grid_id,
                    grid_index=grid_index,
                    K=self.config_.n_grid_per_axis,
                )
                n_random_samples = self._compute_random_sample_count(
                    len(neighbor_train_indices)
                )
                random_train_indices = sample_non_neighbor_train_indices(
                    grid_id=grid_id,
                    grid_index=grid_index,
                    K=self.config_.n_grid_per_axis,
                    n_samples=n_random_samples,
                    rng=rng,
                )
                local_train_indices = merge_unique_indices(
                    neighbor_train_indices, random_train_indices
                )

                if not local_train_indices:
                    raise _EmptyContextError(
                        f"No training samples are available for grid {grid_id}. "
                        "Increase s above 0 to include distant samples, or reduce K "
                        "to use a wider neighborhood."
                    )

                local_estimator = self._fit_estimator(
                    self.X_train_.iloc[local_train_indices],
                    self.y_train_.iloc[local_train_indices],
                )
                predictions = local_estimator.predict(
                    X_df.iloc[grid_cell.test_indices][self.model_columns_]
                )
                if hasattr(predictions, "detach"):
                    predictions = predictions.detach().cpu().double().numpy()
                predictions = np.asarray(predictions, dtype=float)
                if predictions.ndim == 2 and predictions.shape[1] == 1:
                    predictions = predictions[:, 0]
                if predictions.ndim != 1 or len(predictions) != len(grid_cell.test_indices):
                    raise ValueError("The model must return one prediction per input row.")
                if not np.isfinite(predictions).all():
                    raise ValueError("The model returned NaN or infinite predictions.")
                ensemble_predictions[ensemble_index, grid_cell.test_indices] = predictions
                train_sizes.append(len(local_train_indices))
                neighbor_sizes.append(len(neighbor_train_indices))

        diagnostics = PredictionDiagnostics(
            average_train_size=float(np.mean(train_sizes)) if train_sizes else 0.0,
            average_neighbor_size=(
                float(np.mean(neighbor_sizes)) if neighbor_sizes else 0.0
            ),
            fitted_local_models=len(train_sizes),
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
            return 0
        remaining_train_pool = max(0, len(self.X_train_) - n_neighbor_samples)
        return max(
            self.config_.min_random_samples, int(remaining_train_pool * self.config_.s)
        )

    def _validate_columns(self, X: pd.DataFrame) -> None:
        missing_columns = [
            column for column in self.spa_cols if column not in X.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required spatial columns: {missing_columns}")

        if len(self.spa_cols) != 2:
            raise ValueError("Exactly two spa columns are required.")
        if len(set(self.spa_cols)) != 2:
            raise ValueError("spa_cols must contain two distinct column names.")

        try:
            coordinates = X[self.spa_cols].to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("Spatial coordinates must be numeric and finite.") from error
        if not np.isfinite(coordinates).all():
            raise ValueError("Spatial coordinates must not contain missing or infinite values.")
        X[self.spa_cols] = coordinates

        if self.x_cols is not None:
            feature_missing = [
                column for column in self.x_cols if column not in X.columns
            ]
            if feature_missing:
                raise ValueError(f"Missing required x columns: {feature_missing}")
            if len(set(self.x_cols)) != len(self.x_cols):
                raise ValueError("x_cols must not contain duplicate columns.")
            overlap = set(self.x_cols).intersection(self.spa_cols)
            if overlap:
                raise ValueError(
                    "x_cols and spa_cols must not overlap. Overlapping columns: "
                    f"{sorted(overlap)}"
                )

    def _resolve_spa_bounds(self, train_coords: pd.DataFrame) -> GridBounds:
        if self.spa_bounds is None:
            return build_grid_bounds(train_coords)

        missing_bounds = [
            column for column in self.spa_cols if column not in self.spa_bounds
        ]
        if missing_bounds:
            raise ValueError(f"Missing spa_bounds entries for: {missing_bounds}")

        mins: list[float] = []
        maxs: list[float] = []
        for column in self.spa_cols:
            try:
                lower, upper = self.spa_bounds[column]
                lower, upper = float(lower), float(upper)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"spa_bounds for {column!r} must be a finite (lower, upper) pair."
                ) from error
            if not np.isfinite([lower, upper]).all() or upper <= lower:
                raise ValueError(
                    f"spa_bounds for {column!r} must be finite with upper > lower."
                )
            mins.append(float(lower))
            maxs.append(float(upper))

        return GridBounds(
            mins=np.asarray(mins, dtype=float),
            maxs=np.asarray(maxs, dtype=float),
        )

    @staticmethod
    def _validate_dataframe(X: Any, variable_name: str) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{variable_name} must be a pandas DataFrame.")
        if X.empty:
            raise ValueError(f"{variable_name} must contain at least one row and column.")
        if not X.columns.is_unique:
            raise ValueError(f"{variable_name} must not contain duplicate column names.")
        return X.copy()

    @staticmethod
    def _validate_target(y: pd.Series | pd.DataFrame | np.ndarray) -> pd.Series:
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("Only a single regression target is supported.")
            y = y.iloc[:, 0]
        if not isinstance(y, pd.Series):
            y_array = np.asarray(y)
            if y_array.ndim == 2 and y_array.shape[1] == 1:
                y_array = y_array[:, 0]
            if y_array.ndim != 1:
                raise ValueError("Only a single regression target is supported.")
            y = pd.Series(y_array, name="target")
        try:
            target = y.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError("y must contain numeric, finite target values.") from error
        if not np.isfinite(target.to_numpy()).all():
            raise ValueError("y must not contain missing or infinite values.")
        return target
