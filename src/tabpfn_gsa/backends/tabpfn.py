from __future__ import annotations

from typing import Any

from tabpfn_gsa.backends.base import (
    BuiltModelBackend,
    ModelRuntimeInfo,
    detect_local_devices,
    extract_requested_device,
    safe_package_version,
    select_local_device,
)


def build_local_tabpfn_backend(
    model_kwargs: dict[str, Any],
    device: str = "auto",
) -> BuiltModelBackend:
    """Build a local sklearn-compatible TabPFN estimator for the GSA core."""

    model_kwargs = dict(model_kwargs)
    model_kwargs.setdefault("device", select_local_device(device))
    estimator = _build_local_tabpfn_estimator(model_kwargs=model_kwargs)

    runtime_info = ModelRuntimeInfo(
        backend_name="tabpfn",
        requested_execution="local",
        resolved_execution="local",
        estimator_class=type(estimator).__name__,
        package_name="tabpfn",
        package_version=safe_package_version("tabpfn"),
        model_version=_format_tabpfn_model_version(model_kwargs=model_kwargs),
        torch_package_version=safe_package_version("torch"),
        requested_device=extract_requested_device(model_kwargs),
        available_local_devices=detect_local_devices(),
        sends_data_to_remote=False,
        phe_enabled=False,
        model_kwargs=model_kwargs,
        notes=(
            "Default backend uses the local open-source tabpfn Python package.",
            "Local TabPFN supports CUDA and recent releases also support Apple MPS.",
            "Recent local versions may require gated Hugging Face model access on first use.",
            "PHE is intentionally not supported in this simplified project build.",
        ),
    )
    return BuiltModelBackend(estimator=estimator, runtime_info=runtime_info)


def _build_local_tabpfn_estimator(
    model_kwargs: dict[str, Any],
) -> Any:
    if "_tabpfn_version" in model_kwargs:
        raise ValueError(
            "Do not use _tabpfn_version. GSAModel uses the default model from the installed tabpfn package."
        )

    regressor_class = _import_tabpfn_regressor()
    return regressor_class(**model_kwargs)


def _import_tabpfn_regressor() -> Any:
    try:
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor
    except Exception:
        try:
            import tabpfn.regressor as tabpfn_regressor

            return tabpfn_regressor.TabPFNRegressor
        except Exception as error:
            raise ImportError(
                "Could not import TabPFNRegressor. Install local TabPFN with "
                "`pip install -e .`."
            ) from error


def _format_tabpfn_model_version(model_kwargs: dict[str, Any]) -> str:
    if "model_path" in model_kwargs:
        return f'model_path={model_kwargs["model_path"]}'
    return "package default"
