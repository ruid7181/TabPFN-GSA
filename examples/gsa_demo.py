from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from tabpfn_gsa import GSAModel


def make_synthetic_geospatial_dataset(n_samples: int = 400, random_state: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    coord_x = rng.uniform(0.0, 1.0, size=n_samples)
    coord_y = rng.uniform(0.0, 1.0, size=n_samples)
    x1 = rng.normal(size=n_samples)
    x2 = rng.normal(size=n_samples)
    target = x1 + 0.5 * x2 + np.sin(coord_x * 6.0) - np.cos(coord_y * 4.0) + rng.normal(scale=0.05, size=n_samples)

    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "coord_x": coord_x,
            "coord_y": coord_y,
            "target": target,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a GSA demo with TabPFN.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ignore-pretraining-limits", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = make_synthetic_geospatial_dataset()
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=0)

    model_kwargs = {}
    model_kwargs["ignore_pretraining_limits"] = args.ignore_pretraining_limits

    model = GSAModel(
        spa_cols=["coord_x", "coord_y"],
        x_cols=["x1", "x2"],
        device=args.device,
        verbose=args.verbose,
        K=64,
        s=0.1,
        random_state=0,
        model_kwargs=model_kwargs,
    )

    X_train = train_df[["x1", "x2", "coord_x", "coord_y"]]
    y_train = train_df["target"]
    X_test = test_df[["x1", "x2", "coord_x", "coord_y"]]
    y_test = test_df["target"].to_numpy()

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if not args.verbose:
        print(model.format_runtime_info())
    print(f"R2: {r2_score(y_test, predictions):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, predictions):.4f}")


if __name__ == "__main__":
    main()
