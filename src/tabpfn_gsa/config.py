from dataclasses import dataclass
from math import isqrt


@dataclass(frozen=True)
class GSAConfig:
    """Configuration for the grid-based geospatial sparse attention workflow."""

    K: int = 64
    s: float = 0.1
    n_ensembles: int = 3
    min_random_samples: int = 2
    include_spatial_features: bool = True
    use_global_fallback: bool = True

    def __post_init__(self) -> None:
        if self.K < 1:
            raise ValueError("K must be at least 1.")
        if isqrt(self.K) ** 2 != self.K:
            raise ValueError("K must be a square number, for example 4, 9, 16, 25, or 64.")
        if not 0.0 <= self.s <= 1.0:
            raise ValueError("s must be between 0 and 1.")
        if self.n_ensembles < 1:
            raise ValueError("n_ensembles must be at least 1.")
        if self.min_random_samples < 0:
            raise ValueError("min_random_samples must be non-negative.")

    @property
    def n_grid_per_axis(self) -> int:
        """Number of grid cells per coordinate axis."""

        return isqrt(self.K)
