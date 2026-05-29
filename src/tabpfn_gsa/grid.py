from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

GridId = tuple[int, int]


@dataclass(frozen=True)
class GridCell:
    """Index lists for one grid cell."""

    train_indices: list[int]
    test_indices: list[int]


def build_regular_grid_index(
    train_coords: pd.DataFrame,
    test_coords: pd.DataFrame,
    K: int,
) -> dict[GridId, GridCell]:
    """Build a regular 2D grid and assign train and test rows to cells."""

    combined = np.vstack([train_coords.to_numpy(), test_coords.to_numpy()])
    mins = combined.min(axis=0)
    spans = combined.max(axis=0) - mins
    spans = np.where(spans < 1e-12, 1.0, spans)
    steps = spans / K

    grid_index: dict[GridId, GridCell] = {
        (i, j): GridCell(train_indices=[], test_indices=[])
        for i in range(K)
        for j in range(K)
    }

    train_grid_ids = _assign_grid_ids(train_coords.to_numpy(), mins, steps, K)
    test_grid_ids = _assign_grid_ids(test_coords.to_numpy(), mins, steps, K)

    mutable_grid_index = {
        grid_id: {"train_indices": list(cell.train_indices), "test_indices": list(cell.test_indices)}
        for grid_id, cell in grid_index.items()
    }

    for row_index, grid_id in enumerate(train_grid_ids):
        mutable_grid_index[grid_id]["train_indices"].append(row_index)

    for row_index, grid_id in enumerate(test_grid_ids):
        mutable_grid_index[grid_id]["test_indices"].append(row_index)

    return {
        grid_id: GridCell(
            train_indices=payload["train_indices"],
            test_indices=payload["test_indices"],
        )
        for grid_id, payload in mutable_grid_index.items()
    }


def collect_neighbor_train_indices(
    grid_id: GridId,
    grid_index: dict[GridId, GridCell],
    K: int,
) -> list[int]:
    """Collect training indices from the surrounding 3x3 neighborhood."""

    neighbors: list[int] = []
    for neighbor_id in iter_neighbor_grid_ids(grid_id, K):
        neighbors.extend(grid_index[neighbor_id].train_indices)
    return _deduplicate_preserving_order(neighbors)


def sample_non_neighbor_train_indices(
    grid_id: GridId,
    grid_index: dict[GridId, GridCell],
    K: int,
    n_samples: int,
    rng: np.random.Generator,
) -> list[int]:
    """Sample training indices from all non-neighboring cells."""

    if n_samples <= 0:
        return []

    neighbor_ids = set(iter_neighbor_grid_ids(grid_id, K))
    candidates: list[int] = []
    for candidate_grid_id, grid_cell in grid_index.items():
        if candidate_grid_id not in neighbor_ids:
            candidates.extend(grid_cell.train_indices)

    candidates = _deduplicate_preserving_order(candidates)
    if not candidates:
        return []

    if len(candidates) <= n_samples:
        return candidates

    sampled = rng.choice(candidates, size=n_samples, replace=False)
    return sampled.tolist()


def merge_unique_indices(primary: Iterable[int], secondary: Iterable[int]) -> list[int]:
    """Merge two index collections without duplicates while keeping order stable."""

    return _deduplicate_preserving_order([*primary, *secondary])


def iter_neighbor_grid_ids(grid_id: GridId, K: int) -> Iterable[GridId]:
    """Yield valid neighbor ids for a 2D Moore neighborhood."""

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            x = grid_id[0] + dx
            y = grid_id[1] + dy
            if 0 <= x < K and 0 <= y < K:
                yield (x, y)


def _assign_grid_ids(
    coords: np.ndarray,
    mins: np.ndarray,
    steps: np.ndarray,
    K: int,
) -> list[GridId]:
    scaled = np.floor((coords - mins) / steps).astype(int)
    clipped = np.clip(scaled, 0, K - 1)
    return [tuple(cell.tolist()) for cell in clipped]


def _deduplicate_preserving_order(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    unique_values: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    return unique_values
