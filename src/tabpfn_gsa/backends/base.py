from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any


@dataclass(frozen=True)
class ModelRuntimeInfo:
    """Runtime metadata reported by a model backend."""

    backend_name: str
    requested_execution: str
    resolved_execution: str
    estimator_class: str
    package_name: str
    package_version: str | None
    model_version: str | None
    torch_package_version: str | None
    requested_device: str | None
    available_local_devices: tuple[str, ...]
    sends_data_to_remote: bool
    phe_enabled: bool
    model_kwargs: dict[str, Any]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class BuiltModelBackend:
    """A sklearn-compatible estimator plus runtime metadata for GSAModel."""

    estimator: Any
    runtime_info: ModelRuntimeInfo


def format_runtime_info(info: ModelRuntimeInfo) -> str:
    model_version = info.model_version or "user-managed"
    requested_device = info.requested_device or "default"
    available_devices = ", ".join(info.available_local_devices) if info.available_local_devices else "n/a"
    lines = [
        f'{" GSA Runtime ":=^60}',
        f'{"backend":<24}{info.backend_name}',
        f'{"execution":<24}{info.resolved_execution}',
        f'{"package":<24}{info.package_name}',
        f'{"package version":<24}{info.package_version or "unknown"}',
        f'{"model version":<24}{model_version}',
        f'{"PyTorch version":<24}{info.torch_package_version or "n/a"}',
        f'{"device":<24}{requested_device}',
        f'{"available devices":<24}{available_devices}',
        f'{"remote data transfer":<24}{"yes" if info.sends_data_to_remote else "no"}',
        f'{"PHE":<24}{"enabled" if info.phe_enabled else "not enabled"}',
        "=" * 60,
    ]
    return "\n".join(lines)


def safe_package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def extract_requested_device(model_kwargs: dict[str, Any]) -> str | None:
    device = model_kwargs.get("device")
    if device is None:
        return None
    return str(device)


def detect_local_devices() -> tuple[str, ...]:
    devices = ["cpu"]
    try:
        import torch
    except Exception:
        return tuple(devices)

    if torch.cuda.is_available():
        devices.insert(0, "cuda")

    try:
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            devices.append("mps")
    except Exception:
        pass

    return tuple(devices)


def select_local_device(device: str = "auto") -> str:
    """Resolve the local accelerator preference used by local model backends."""

    if device != "auto":
        return device

    devices = detect_local_devices()
    if "cuda" in devices:
        return "cuda"
    if "mps" in devices:
        return "mps"
    return "cpu"
