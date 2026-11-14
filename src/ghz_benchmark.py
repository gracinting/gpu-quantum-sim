"""GHZ statevector benchmark: CUDA-Q on CPU (``qpp-cpu``) vs GPU (``nvidia``).

For each qubit count the script prepares an n-qubit GHZ state, samples it, and
records the median wall time over several runs. Two guards keep a sweep from
taking down the machine: a memory estimate (2**n amplitudes) checked against
free RAM / VRAM, and a projected-time budget per data point.

Usage::

    python src/ghz_benchmark.py                 # full sweep, n = 20..30
    python src/ghz_benchmark.py --quick         # < 2 minutes on a laptop CPU
    python src/ghz_benchmark.py --plot-only     # redraw plots from the CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cudaq
from matplotlib.ticker import FuncFormatter, LogFormatter, MaxNLocator

import hardware
import plotting

logger = logging.getLogger("ghz_benchmark")

CSV_NAME = "ghz_benchmark.csv"
META_NAME = "ghz_benchmark_meta.json"
RUNTIME_PLOT_NAME = "ghz_benchmark.png"
SPEEDUP_PLOT_NAME = "ghz_speedup.png"
CSV_FIELDS = ("target", "n", "median_s", "std_s", "est_mem_gb")

CPU_TARGET = "qpp-cpu"
GPU_TARGET = "nvidia"
RANDOM_SEED = 42

#: Runtime grows at least 2x per added qubit (the state doubles). Used to
#: project the cost of the next point before we have two measurements.
MIN_GROWTH_PER_QUBIT = 2.0


@cudaq.kernel
def ghz(n: int):
    # (|00...0> + |11...1>) / sqrt(2): one Hadamard, then a CNOT chain.
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(n - 1):
        x.ctrl(q[i], q[i + 1])
    mz(q)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TargetSpec:
    """A CUDA-Q simulation target and how to select it."""

    name: str
    device: str  # "cpu" or "gpu"
    label: str
    options: dict[str, str] = field(default_factory=dict)


TARGETS: dict[str, TargetSpec] = {
    CPU_TARGET: TargetSpec(CPU_TARGET, "cpu", "CPU (qpp-cpu)"),
    GPU_TARGET: TargetSpec(GPU_TARGET, "gpu", "GPU (nvidia, cuStateVec)", {"option": "fp32"}),
}


@dataclass(frozen=True)
class BenchmarkRow:
    """One measured (target, qubit count) point. Mirrors a CSV row."""

    target: str
    n: int
    median_s: float
    std_s: float
    est_mem_gb: float


@dataclass(frozen=True)
class SkippedPoint:
    """A (target, qubit count) point that was deliberately not run."""

    target: str
    n: int | None
    reason: str


@dataclass
class BenchmarkReport:
    """Everything a benchmark run produced."""

    rows: list[BenchmarkRow]
    skipped: list[SkippedPoint]
    meta: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkConfig:
    """Parameters of a benchmark sweep."""

    min_qubits: int = 20
    max_qubits: int = 30
    shots: int = 1000
    warmup: int = 1
    repeats: int = 5
    targets: tuple[str, ...] = (CPU_TARGET, GPU_TARGET)
    time_budget_s: float = 600.0
    safety_factor: float = hardware.DEFAULT_SAFETY_FACTOR
    quick: bool = False

    def __post_init__(self) -> None:
        if self.min_qubits < 2:
            raise ValueError("min_qubits must be at least 2")
        if self.max_qubits < self.min_qubits:
            raise ValueError("max_qubits must be >= min_qubits")
        if self.repeats < 1 or self.warmup < 0 or self.shots < 1:
            raise ValueError("repeats >= 1, warmup >= 0 and shots >= 1 are required")
        unknown = set(self.targets) - set(TARGETS)
        if unknown:
            raise ValueError(f"unknown targets: {sorted(unknown)}")


# --------------------------------------------------------------------------- #
# Correctness
# --------------------------------------------------------------------------- #


def assert_ghz_counts(counts: dict[str, int], n: int, tolerance: float | None = None) -> None:
    """Raise AssertionError unless ``counts`` looks like an n-qubit GHZ state.

    Only the all-zeros and all-ones bitstrings may appear. If ``tolerance`` is
    given, each must also account for 50% of the shots to within it.
    """
    expected = {"0" * n, "1" * n}
    unexpected = set(counts) - expected
    if unexpected:
        raise AssertionError(
            f"GHZ({n}) produced non-GHZ bitstrings: {sorted(unexpected)[:5]}"
        )
    if tolerance is None:
        return
    shots = sum(counts.values())
    for bitstring in sorted(expected):
        fraction = counts.get(bitstring, 0) / shots
        if abs(fraction - 0.5) > tolerance:
            raise AssertionError(
                f"GHZ({n}): P({bitstring[0]}...{bitstring[-1]}) = {fraction:.3f}, "
                f"expected 0.5 +/- {tolerance}"
            )


def check_ghz_correctness(
    n: int = 8, shots: int = 10_000, tolerance: float = 0.05
) -> dict[str, int]:
    """Sample GHZ(n) on the active target and verify the measurement statistics.

    Returns:
        The observed counts, e.g. ``{"00000000": 5012, "11111111": 4988}``.

    Raises:
        AssertionError: If any other bitstring appears or the split is not ~50/50.
    """
    cudaq.set_random_seed(RANDOM_SEED)
    counts = dict(cudaq.sample(ghz, n, shots_count=shots).items())
    assert_ghz_counts(counts, n, tolerance)
    return counts


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #


def activate_target(spec: TargetSpec) -> str:
    """Select ``spec`` as the CUDA-Q target and return its precision.

    The precision is read back from CUDA-Q rather than assumed, because the
    memory estimate depends on it (``qpp-cpu`` is fp64, ``nvidia`` is fp32
    unless asked otherwise).
    """
    cudaq.set_target(spec.name, **spec.options)
    precision = cudaq.get_target().get_precision()
    return "fp32" if precision == cudaq.SimulationPrecision.fp32 else "fp64"


def free_device_memory_bytes(spec: TargetSpec) -> int:
    """Return free RAM for a CPU target or free VRAM for a GPU target.

    Unknown VRAM is reported as 0 so that the caller skips instead of gambling.
    """
    if spec.device == "gpu":
        gpu = hardware.query_nvidia_gpu()
        return gpu.free_bytes if gpu else 0
    return hardware.available_ram_bytes()


def time_ghz(n: int, shots: int, warmup: int, repeats: int) -> list[float]:
    """Time ``repeats`` GHZ(n) simulations on the active target.

    Warm-up runs absorb one-off costs (kernel JIT compilation, CUDA context
    creation) and double as a correctness check on the result.

    Returns:
        Wall-clock seconds for each timed run.
    """
    for _ in range(warmup):
        counts = dict(cudaq.sample(ghz, n, shots_count=shots).items())
        assert_ghz_counts(counts, n)
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        cudaq.sample(ghz, n, shots_count=shots)
        timings.append(time.perf_counter() - start)
    return timings


def project_point_seconds(rows: Sequence[BenchmarkRow], n: int, runs: int) -> float | None:
    """Estimate the wall time of measuring ``n`` qubits from earlier points.

    Extrapolates the last measured median using the observed per-qubit growth
    (never less than 2x, since the state doubles with each qubit).

    Returns:
        Projected seconds for ``runs`` executions, or None with no history.
    """
    if not rows:
        return None
    last = rows[-1]
    growth = MIN_GROWTH_PER_QUBIT
    if len(rows) >= 2:
        previous = rows[-2]
        span = last.n - previous.n
        if span > 0 and previous.median_s > 0:
            growth = max(growth, (last.median_s / previous.median_s) ** (1.0 / span))
    return last.median_s * growth ** (n - last.n) * runs


def benchmark_target(
    spec: TargetSpec, config: BenchmarkConfig
) -> tuple[list[BenchmarkRow], list[SkippedPoint], str]:
    """Sweep qubit counts on one target.

    Returns:
        Measured rows, skipped points with reasons, and the target precision.
    """
    precision = activate_target(spec)
    cudaq.set_random_seed(RANDOM_SEED)
    memory_kind = "VRAM" if spec.device == "gpu" else "RAM"
    runs_per_point = config.warmup + config.repeats
    rows: list[BenchmarkRow] = []
    skipped: list[SkippedPoint] = []
    logger.info("Target %s (%s, %s)", spec.name, spec.device.upper(), precision)

    for n in range(config.min_qubits, config.max_qubits + 1):
        required = hardware.estimate_statevector_bytes(n, precision)
        required_gib = hardware.bytes_to_gib(required)
        free = free_device_memory_bytes(spec)
        if not hardware.fits_in_memory(required, free, config.safety_factor):
            reason = (
                f"statevector needs {required_gib:.2f} GiB but only "
                f"{hardware.bytes_to_gib(free):.2f} GiB {memory_kind} is free "
                f"(using at most {config.safety_factor:.0%} of it)"
            )
            logger.warning("  n=%d skipped: %s", n, reason)
            skipped.append(SkippedPoint(spec.name, n, reason))
            continue

        projected = project_point_seconds(rows, n, runs_per_point)
        if config.time_budget_s > 0 and projected is not None and projected > config.time_budget_s:
            reason = (
                f"projected {projected:.0f} s for {runs_per_point} runs exceeds the "
                f"{config.time_budget_s:.0f} s per-point time budget"
            )
            logger.warning("  n=%d skipped: %s", n, reason)
            skipped.append(SkippedPoint(spec.name, n, reason))
            continue

        try:
            timings = time_ghz(n, config.shots, config.warmup, config.repeats)
        except (RuntimeError, MemoryError) as exc:
            reason = f"simulator raised {type(exc).__name__}: {exc}"
            logger.error("  n=%d failed, stopping this target: %s", n, reason)
            skipped.append(SkippedPoint(spec.name, n, reason))
            break

        row = BenchmarkRow(
            target=spec.name,
            n=n,
            median_s=statistics.median(timings),
            std_s=statistics.stdev(timings) if len(timings) > 1 else 0.0,
            est_mem_gb=required_gib,
        )
        rows.append(row)
        logger.info(
            "  n=%d  median %.4f s  std %.4f s  (%.3f GiB)",
            n,
            row.median_s,
            row.std_s,
            row.est_mem_gb,
        )
    return rows, skipped, precision


def run_benchmark(config: BenchmarkConfig) -> BenchmarkReport:
    """Run the configured sweep on every requested target that exists here."""
    rows: list[BenchmarkRow] = []
    skipped: list[SkippedPoint] = []
    precisions: dict[str, str] = {}
    gpu_available = hardware.gpu_target_available()

    for name in config.targets:
        spec = TARGETS[name]
        if spec.device == "gpu" and not gpu_available:
            reason = (
                "no NVIDIA GPU visible to CUDA-Q on this machine, so the 'nvidia' "
                "target was skipped. Run notebooks/colab_quickstart.ipynb on a "
                "GPU runtime for the GPU half of the benchmark."
            )
            logger.warning("GPU benchmark skipped: %s", reason)
            skipped.append(SkippedPoint(spec.name, None, reason))
            continue
        target_rows, target_skipped, precision = benchmark_target(spec, config)
        rows.extend(target_rows)
        skipped.extend(target_skipped)
        precisions[name] = precision

    cudaq.reset_target()
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(config),
        "precision": precisions,
        "hardware": hardware.describe_hardware(),
        "skipped": [asdict(point) for point in skipped],
    }
    return BenchmarkReport(rows=rows, skipped=skipped, meta=meta)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def write_csv(rows: Sequence[BenchmarkRow], path: Path) -> Path:
    """Write benchmark rows to ``path`` using the documented column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "target": row.target,
                    "n": row.n,
                    "median_s": f"{row.median_s:.6f}",
                    "std_s": f"{row.std_s:.6f}",
                    # Significant digits, not fixed decimals: small states are
                    # far below 1 GiB and must not be rounded away.
                    "est_mem_gb": f"{row.est_mem_gb:.9g}",
                }
            )
    return path


def read_csv(path: Path) -> list[BenchmarkRow]:
    """Read benchmark rows previously written by :func:`write_csv`."""
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            BenchmarkRow(
                target=record["target"],
                n=int(record["n"]),
                median_s=float(record["median_s"]),
                std_s=float(record["std_s"]),
                est_mem_gb=float(record["est_mem_gb"]),
            )
            for record in csv.DictReader(handle)
        ]


def rows_for(rows: Sequence[BenchmarkRow], target: str) -> list[BenchmarkRow]:
    """Return the rows of one target, sorted by qubit count."""
    return sorted((row for row in rows if row.target == target), key=lambda row: row.n)


def compute_speedups(rows: Sequence[BenchmarkRow]) -> list[tuple[int, float]]:
    """Return ``(n, cpu_time / gpu_time)`` for every n measured on both targets."""
    cpu = {row.n: row.median_s for row in rows_for(rows, CPU_TARGET)}
    gpu = {row.n: row.median_s for row in rows_for(rows, GPU_TARGET)}
    return [(n, cpu[n] / gpu[n]) for n in sorted(cpu.keys() & gpu.keys()) if gpu[n] > 0]


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def format_seconds(seconds: float) -> str:
    """Format a duration compactly for a data label."""
    if seconds < 1.0:
        return f"{seconds * 1e3:.0f} ms"
    if seconds < 100.0:
        return f"{seconds:.1f} s"
    return f"{seconds:.0f} s"


def _hardware_subtitle(meta: dict[str, Any]) -> str:
    """Describe the run conditions in one line for a plot subtitle."""
    config = meta.get("config", {})
    hw = meta.get("hardware", {})
    parts = []
    if config:
        parts.append(
            f"{config.get('shots')} shots, median of {config.get('repeats')} runs "
            f"after {config.get('warmup')} warm-up"
        )
    devices = [hw.get("cpu")]
    if hw.get("gpu"):
        devices.append(hw["gpu"]["name"])
    devices = [device for device in devices if device]
    if devices:
        parts.append(" / ".join(devices))
    return "  |  ".join(parts)


def plot_runtime(rows: Sequence[BenchmarkRow], meta: dict[str, Any], path: Path) -> Path:
    """Plot median runtime against qubit count on a log axis, one line per target."""
    plotting.apply_style()
    fig, ax = plotting.plt.subplots(figsize=(8.0, 5.0))
    precision = meta.get("precision", {})
    series = [
        (TARGETS[CPU_TARGET], plotting.SERIES_1, "o"),
        (TARGETS[GPU_TARGET], plotting.SERIES_2, "s"),
    ]
    drawn = 0
    for spec, colour, marker in series:
        target_rows = rows_for(rows, spec.name)
        if not target_rows:
            continue
        drawn += 1
        xs = [row.n for row in target_rows]
        ys = [row.median_s for row in target_rows]
        label = spec.label
        if spec.name in precision:
            label = f"{label[:-1]}, {precision[spec.name]})"
        ax.errorbar(
            xs,
            ys,
            yerr=[row.std_s for row in target_rows],
            color=colour,
            marker=marker,
            markeredgecolor=plotting.SURFACE,
            markeredgewidth=1.5,
            ecolor=colour,
            elinewidth=1.2,
            label=label,
            zorder=3,
        )
        ax.annotate(
            format_seconds(ys[-1]),
            (xs[-1], ys[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            color=plotting.INK_SECONDARY,
        )
    ax.set_yscale("log")
    # Plain numbers ("20", "0.5") instead of "2 x 10^1": easier to read as seconds.
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_formatter(LogFormatter(labelOnlyBase=False, minor_thresholds=(1.5, 0.4)))
    ax.set_xlabel("Number of qubits, n  (statevector holds $2^n$ amplitudes)")
    ax.set_ylabel("Median wall time per run (s)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.margins(x=0.08)
    ax.grid(axis="x", visible=False)
    plotting.set_title(
        ax, "GHZ statevector simulation: runtime vs. qubit count", _hardware_subtitle(meta)
    )
    # The legend stays even for one series: it is the only place the target
    # name and its precision appear.
    ax.legend(loc="upper left")
    if drawn < 2 and not rows_for(rows, GPU_TARGET):
        fig.text(
            0.125,
            -0.02,
            "GPU target not shown: no NVIDIA GPU on the machine that produced this run.",
            fontsize=9,
            color=plotting.MUTED,
            ha="left",
        )
    return plotting.save_figure(fig, path)


def plot_speedup(rows: Sequence[BenchmarkRow], meta: dict[str, Any], path: Path) -> Path | None:
    """Plot GPU speedup (CPU time / GPU time) against qubit count.

    Returns:
        The written path, or None when there is no n measured on both targets.
    """
    speedups = compute_speedups(rows)
    if not speedups:
        return None
    plotting.apply_style()
    fig, ax = plotting.plt.subplots(figsize=(8.0, 5.0))
    xs = [n for n, _ in speedups]
    ys = [speedup for _, speedup in speedups]
    ax.plot(
        xs,
        ys,
        color=plotting.SERIES_1,
        marker="o",
        markeredgecolor=plotting.SURFACE,
        markeredgewidth=1.5,
        zorder=3,
    )
    ax.axhline(1.0, color=plotting.MUTED, linestyle="--", linewidth=1.2, zorder=2)
    ax.annotate(
        "break-even (1×)",
        (xs[0], 1.0),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=9,
        color=plotting.MUTED,
    )
    ax.annotate(
        f"{ys[-1]:.0f}×" if ys[-1] >= 10 else f"{ys[-1]:.1f}×",
        (xs[-1], ys[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=plotting.INK,
    )
    if max(ys) / max(min(ys), 1e-12) > 20:
        ax.set_yscale("log")
    ax.set_xlabel("Number of qubits, n")
    ax.set_ylabel("Speedup (CPU time / GPU time)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.margins(x=0.08)
    ax.grid(axis="x", visible=False)
    plotting.set_title(
        ax, "GPU speedup over CPU for GHZ simulation", _hardware_subtitle(meta)
    )
    return plotting.save_figure(fig, path)


def make_plots(rows: Sequence[BenchmarkRow], meta: dict[str, Any], out_dir: Path) -> list[Path]:
    """Write every plot that the available data supports."""
    if not rows:
        logger.warning("No benchmark rows to plot.")
        return []
    written = [plot_runtime(rows, meta, out_dir / RUNTIME_PLOT_NAME)]
    speedup_path = plot_speedup(rows, meta, out_dir / SPEEDUP_PLOT_NAME)
    if speedup_path is None:
        logger.warning(
            "Speedup plot skipped: it needs at least one qubit count measured on "
            "both the CPU and the GPU target."
        )
    else:
        written.append(speedup_path)
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Benchmark GHZ statevector simulation on CUDA-Q CPU and GPU targets."
    )
    parser.add_argument("--min-qubits", type=int, default=None, help="smallest n (default 20)")
    parser.add_argument("--max-qubits", type=int, default=None, help="largest n (default 30)")
    parser.add_argument("--shots", type=int, default=1000, help="shots per run (default 1000)")
    parser.add_argument("--repeats", type=int, default=None, help="timed runs per point (default 5)")
    parser.add_argument("--warmup", type=int, default=1, help="untimed warm-up runs (default 1)")
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=sorted(TARGETS),
        default=[CPU_TARGET, GPU_TARGET],
        help="targets to benchmark (default: both)",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=600.0,
        help="skip a point when its projected wall time (all runs) exceeds this "
        "many seconds; 0 disables the guard (default 600)",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="output directory")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="reduced sweep (n = 16..20, 3 repeats) that finishes in under two "
        "minutes on a CPU; writes to results/quick unless --out-dir is given",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="regenerate the plots from an existing CSV without re-running",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    """Resolve CLI arguments (and ``--quick`` defaults) into a config."""
    quick = bool(args.quick)

    def pick(value: int | None, full: int, reduced: int) -> int:
        return value if value is not None else (reduced if quick else full)

    return BenchmarkConfig(
        min_qubits=pick(args.min_qubits, 20, 16),
        max_qubits=pick(args.max_qubits, 30, 20),
        shots=args.shots,
        warmup=args.warmup,
        repeats=pick(args.repeats, 5, 3),
        targets=tuple(args.targets),
        time_budget_s=args.time_budget,
        quick=quick,
    )


def default_out_dir(quick: bool) -> Path:
    """Return ``results/`` (or ``results/quick/``) next to the repo root."""
    results = Path(__file__).resolve().parent.parent / "results"
    return results / "quick" if quick else results


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = args.out_dir or default_out_dir(args.quick)
    csv_path = out_dir / CSV_NAME
    meta_path = out_dir / META_NAME

    if args.plot_only:
        if not csv_path.exists():
            logger.error("Nothing to plot: %s does not exist. Run the benchmark first.", csv_path)
            return 1
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        for path in make_plots(read_csv(csv_path), meta, out_dir):
            logger.info("Wrote %s", path)
        return 0

    config = config_from_args(args)

    cudaq.set_target(CPU_TARGET)
    counts = check_ghz_correctness()
    logger.info("Correctness check passed on %s: GHZ(8) -> %s", CPU_TARGET, counts)

    report = run_benchmark(config)
    if not report.rows:
        logger.error("No data points were measured; nothing written.")
        return 1

    write_csv(report.rows, csv_path)
    meta_path.write_text(json.dumps(report.meta, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", csv_path)
    logger.info("Wrote %s", meta_path)
    for path in make_plots(report.rows, report.meta, out_dir):
        logger.info("Wrote %s", path)

    largest = max(report.rows, key=lambda row: row.n)
    logger.info(
        "Largest simulation: %d qubits on %s (%.3f GiB, median %.3f s)",
        largest.n,
        largest.target,
        largest.est_mem_gb,
        largest.median_s,
    )
    speedups = compute_speedups(report.rows)
    if speedups:
        n, speedup = speedups[-1]
        logger.info("GPU speedup at n=%d: %.1fx", n, speedup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
