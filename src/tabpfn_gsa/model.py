from __future__ import annotations

from typing import Any

from tabpfn_gsa.backends.base import ModelRuntimeInfo, format_runtime_info
from tabpfn_gsa.backends.functions import build_function_backend
from tabpfn_gsa.backends.tabpfn import build_local_tabpfn_backend
from tabpfn_gsa.estimator import GSARegressor


class GSAModel(GSARegressor):
    """Unified GSA entrypoint for supported tabular foundation models."""

    def __init__(
        self,
        spa_cols: list[str] | None = None,
        x_cols: list[str] | None = None,
        K: int = 64,
        s: float = 0.1,
        random_state: int | None = 0,
        device: str = "auto",
        verbose: bool = False,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if spa_cols is None:
            raise ValueError("spa_cols is required.")

        self.spa_cols = spa_cols
        self.x_cols = x_cols
        self.K = K
        self.s = s
        self.n_ensembles = 3
        self.min_random_samples = 2
        self.include_spatial_features = True
        self.use_global_fallback = True
        self.random_state = random_state
        self.device = device
        self.verbose = verbose
        self.model_kwargs = model_kwargs

        backend = self._build_backend()
        self.runtime_info_ = backend.runtime_info
        self.resolved_execution = backend.runtime_info.resolved_execution

        super().__init__(
            base_estimator=backend.estimator,
            spa_cols=spa_cols,
            x_cols=x_cols,
            K=K,
            s=s,
            n_ensembles=self.n_ensembles,
            min_random_samples=self.min_random_samples,
            include_spatial_features=self.include_spatial_features,
            use_global_fallback=self.use_global_fallback,
            random_state=self.random_state,
            verbose=verbose,
        )

    def fit(self, X, y):
        if self.verbose:
            print(self.format_runtime_info())
        return super().fit(X, y)

    def get_runtime_info(self) -> ModelRuntimeInfo:
        return self.runtime_info_

    def format_runtime_info(self) -> str:
        return format_runtime_info(self.runtime_info_)

    def _build_backend(self):
        kwargs = dict(self.model_kwargs or {})

        if "fit_fn" in kwargs or "predict_fn" in kwargs:
            return build_function_backend(model_kwargs=kwargs)

        return build_local_tabpfn_backend(
            model_kwargs=kwargs,
            device=self.device,
        )
