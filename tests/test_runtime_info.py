from __future__ import annotations

from tabpfn_gsa.backends.base import ModelRuntimeInfo, format_runtime_info


def test_runtime_info_formats_generic_model_metadata() -> None:
    info = ModelRuntimeInfo(
        backend_name="example-backend",
        requested_execution="local",
        resolved_execution="local",
        estimator_class="ExampleRegressor",
        package_name="example-package",
        package_version="1.0.0",
        model_version="1.0.0",
        torch_package_version=None,
        requested_device="cpu",
        available_local_devices=("cpu",),
        sends_data_to_remote=False,
        phe_enabled=False,
        model_kwargs={"device": "cpu"},
        notes=("Example runner for tests.",),
    )

    runtime_text = format_runtime_info(info)

    assert "example-backend" in runtime_text
    assert "local" in runtime_text
    assert "PHE" in runtime_text
    assert "model kwargs" not in runtime_text


def test_runtime_info_keeps_generic_model_metadata() -> None:
    info = ModelRuntimeInfo(
        backend_name="tabpfn",
        requested_execution="local",
        resolved_execution="local",
        estimator_class="TabPFNRegressor",
        package_name="tabpfn",
        package_version="2.5.0",
        model_version="2.5.0",
        torch_package_version="2.0.0",
        requested_device=None,
        available_local_devices=("cpu",),
        sends_data_to_remote=False,
        phe_enabled=False,
        model_kwargs={"ignore_pretraining_limits": True},
        notes=(),
    )

    assert info.package_name == "tabpfn"
    assert info.package_version == "2.5.0"
    assert info.model_version == "2.5.0"
    assert info.model_kwargs == {"ignore_pretraining_limits": True}
