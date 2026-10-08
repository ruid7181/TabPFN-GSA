from __future__ import annotations

import pytest


pytest.importorskip("tabpfn")

from tabpfn_gsa import GSAModel


def test_unified_model_constructs_local_tabpfn_estimator() -> None:
    model = GSAModel(
        spa_cols=["coord_x", "coord_y"],
        x_cols=["x1", "x2"],
        K=4,
        s=0.1,
        random_state=0,
        model_kwargs={"ignore_pretraining_limits": True},
    )

    assert type(model.base_estimator).__name__ == "TabPFNRegressor"
    runtime_info = model.get_runtime_info()
    assert runtime_info.resolved_execution == "local"
    assert runtime_info.package_name == "tabpfn"
    assert runtime_info.phe_enabled is False
    assert "cpu" in runtime_info.available_local_devices


def test_unified_model_runtime_text_exposes_execution_and_phe_state() -> None:
    model = GSAModel(
        spa_cols=["coord_x", "coord_y"],
        x_cols=["x1", "x2"],
        K=4,
        s=0.1,
        random_state=0,
        model_kwargs={"ignore_pretraining_limits": True},
    )

    runtime_text = model.format_runtime_info()

    assert "execution" in runtime_text
    assert "PHE" in runtime_text
    assert "available devices" in runtime_text


def test_unified_model_uses_auto_local_tabpfn_version() -> None:
    model = GSAModel(
        spa_cols=["coord_x", "coord_y"],
        x_cols=["x1", "x2"],
        K=4,
        s=0.1,
        random_state=0,
        model_kwargs={"ignore_pretraining_limits": True},
    )

    runtime_info = model.get_runtime_info()

    assert runtime_info.model_version == "package default"
    assert runtime_info.model_kwargs["device"] in {"cuda", "mps", "cpu"}


def test_refit_refreshes_tabpfn_device_and_seed() -> None:
    import pandas as pd

    model = GSAModel(spa_cols=["coord_x", "coord_y"], device="cpu", random_state=7)
    assert model.base_estimator.random_state == 7
    model.set_params(device="auto", random_state=12)
    X = pd.DataFrame({"coord_x": [0.0, 1.0], "coord_y": [0.0, 1.0]})
    model.fit(X, pd.Series([1.0, 2.0]))
    assert model.base_estimator.random_state == 12
    assert model.get_runtime_info().model_kwargs["random_state"] == 12


def test_explicit_tabpfn_seed_is_preserved() -> None:
    model = GSAModel(
        spa_cols=["coord_x", "coord_y"], random_state=7,
        model_kwargs={"random_state": 99},
    )
    assert model.base_estimator.random_state == 99
