"""Runtime hardware detection and statevector memory estimation.

Everything here is cheap and side-effect free, so it is safe to call before
deciding whether a simulation should be attempted at all. ``cudaq`` is imported
lazily so that the pure-arithmetic helpers (and their tests) work without it.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

import psutil

logger = logging.getLogger(__name__)

#: Bytes used by one complex amplitude at each simulator precision.
#: fp32 -> complex64 (2 x 4 bytes), fp64 -> complex128 (2 x 8 bytes).
BYTES_PER_AMPLITUDE: dict[str, int] = {"fp32": 8, "fp64": 16}

#: Fraction of the free memory we allow a statevector to occupy. Simulators
#: need headroom beyond the state itself (sampling buffers, workspace, the
#: Python process), so we never plan to use 100% of what is free.
DEFAULT_SAFETY_FACTOR: float = 0.8

GIB: int = 1024**3


def estimate_statevector_bytes(n_qubits: int, precision: str = "fp32") -> int:
    """Return the memory needed to store an ``n_qubits`` statevector.

    A statevector holds ``2**n_qubits`` complex amplitudes, each taking
    8 bytes in single precision (complex64) or 16 bytes in double precision
    (complex128).

    Args:
        n_qubits: Number of qubits (non-negative).
        precision: ``"fp32"`` or ``"fp64"``.

    Returns:
        The size of the statevector in bytes.

    Raises:
        ValueError: If ``n_qubits`` is negative or ``precision`` is unknown.
    """
    if n_qubits < 0:
        raise ValueError(f"n_qubits must be non-negative, got {n_qubits}")
    if precision not in BYTES_PER_AMPLITUDE:
        raise ValueError(
            f"precision must be one of {sorted(BYTES_PER_AMPLITUDE)}, got {precision!r}"
        )
    return (1 << n_qubits) * BYTES_PER_AMPLITUDE[precision]


def bytes_to_gib(n_bytes: float) -> float:
    """Convert bytes to gibibytes (2**30 bytes)."""
    return n_bytes / GIB


def fits_in_memory(
    required_bytes: int,
    available_bytes: int,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
) -> bool:
    """Return True if ``required_bytes`` fits in a fraction of the free memory.

    Args:
        required_bytes: Estimated allocation size.
        available_bytes: Memory currently free on the device.
        safety_factor: Fraction of ``available_bytes`` we are willing to use.

    Raises:
        ValueError: If ``safety_factor`` is not in ``(0, 1]``.
    """
    if not 0.0 < safety_factor <= 1.0:
        raise ValueError(f"safety_factor must be in (0, 1], got {safety_factor}")
    return required_bytes <= available_bytes * safety_factor


def available_ram_bytes() -> int:
    """Return the system RAM that can be allocated right now without swapping."""
    return int(psutil.virtual_memory().available)


def total_ram_bytes() -> int:
    """Return the total installed system RAM."""
    return int(psutil.virtual_memory().total)


@dataclass(frozen=True)
class GpuInfo:
    """Name and memory of one NVIDIA GPU, as reported by ``nvidia-smi``."""

    name: str
    total_bytes: int
    free_bytes: int


def query_nvidia_gpu() -> GpuInfo | None:
    """Return details of GPU 0 from ``nvidia-smi``, or None if unavailable."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("nvidia-smi failed: %s", exc)
        return None
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        name, total_mib, free_mib = (field.strip() for field in lines[0].split(","))
        mib = 1024**2
        return GpuInfo(name, int(float(total_mib)) * mib, int(float(free_mib)) * mib)
    except ValueError:
        logger.debug("Could not parse nvidia-smi output: %r", lines[0])
        return None


def gpu_target_available() -> bool:
    """Return True if CUDA-Q can run its ``nvidia`` (cuStateVec) target here.

    Both conditions matter: the installed wheel must ship the target (the
    macOS wheel does not) and CUDA-Q must see at least one CUDA device.
    """
    import cudaq

    try:
        return bool(cudaq.num_available_gpus() > 0 and cudaq.has_target("nvidia"))
    except RuntimeError as exc:
        logger.debug("GPU detection failed: %s", exc)
        return False


def cpu_model() -> str:
    """Return a human-readable CPU model name."""
    system = platform.system()
    try:
        if system == "Darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        if system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except (subprocess.SubprocessError, OSError):
        pass
    return platform.processor() or platform.machine() or "unknown CPU"


def describe_hardware() -> dict[str, Any]:
    """Return a JSON-serialisable description of the machine running the code."""
    import cudaq

    gpu = query_nvidia_gpu()
    return {
        "cpu": cpu_model(),
        "cpu_cores": psutil.cpu_count(logical=False) or psutil.cpu_count() or 0,
        "ram_gib": round(bytes_to_gib(total_ram_bytes()), 1),
        "gpu": asdict(gpu) if gpu else None,
        "gpu_target_available": gpu_target_available(),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "cudaq": cudaq.__version__.split("(")[0].replace("CUDA-Q Version", "").strip(),
    }
