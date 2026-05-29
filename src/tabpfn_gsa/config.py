from dataclasses import dataclass


@dataclass(frozen=True)
class GSAConfig:
    """Configuration for the grid-based geospatial sparse attention workflow."""

    K: int = 10
    s: float = 0.1
    n_ensembles: int = 3
    min_random_samples: int = 2
    include_spatial_features: bool = True
    use_global_fallback: bool = True

    def __post_init__(self) -> None:
        if self.K < 1:
            raise ValueError("K must be at least 1.")
        if not 0.0 <= self.s <= 1.0:
            raise ValueError("s must be between 0 and 1.")
        if self.n_ensembles < 1:
            raise ValueError("n_ensembles must be at least 1.")
        if self.min_random_samples < 0:
            raise ValueError("min_random_samples must be non-negative.")
