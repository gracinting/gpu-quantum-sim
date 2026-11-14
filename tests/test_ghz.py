"""Tests for the GHZ kernel, the benchmark loop and its guards."""

from __future__ import annotations

from pathlib import Path

import cudaq
import pytest

import ghz_benchmark as gb
import hardware


def test_ghz_8_only_all_zeros_and_all_ones() -> None:
    shots = 10_000
    counts = gb.check_ghz_correctness(n=8, shots=shots, tolerance=0.05)
    assert set(counts) == {"0" * 8, "1" * 8}
    assert sum(counts.values()) == shots
    for count in counts.values():
        assert count / shots == pytest.approx(0.5, abs=0.05)


@pytest.mark.parametrize("n", [2, 3, 5, 12])
def test_ghz_is_correct_at_other_sizes(n: int) -> None:
    gb.check_ghz_correctness(n=n, shots=4000, tolerance=0.06)


def test_assert_ghz_counts_rejects_wrong_bitstrings() -> None:
    with pytest.raises(AssertionError, match="non-GHZ"):
        gb.assert_ghz_counts({"000": 500, "111": 490, "010": 10}, 3)


def test_assert_ghz_counts_rejects_a_biased_split() -> None:
    with pytest.raises(AssertionError, match="expected 0.5"):
        gb.assert_ghz_counts({"000": 900, "111": 100}, 3, tolerance=0.05)


def test_time_ghz_returns_one_timing_per_repeat() -> None:
    timings = gb.time_ghz(n=6, shots=100, warmup=1, repeats=4)
    assert len(timings) == 4
    assert all(t > 0 for t in timings)


def test_cpu_target_reports_double_precision() -> None:
    assert gb.activate_target(gb.TARGETS[gb.CPU_TARGET]) == "fp64"


def test_small_cpu_sweep_produces_one_row_per_qubit_count() -> None:
    config = gb.BenchmarkConfig(min_qubits=4, max_qubits=7, shots=100, repeats=3, targets=(gb.CPU_TARGET,))
    report = gb.run_benchmark(config)
    assert [row.n for row in report.rows] == [4, 5, 6, 7]
    assert all(row.target == gb.CPU_TARGET and row.median_s > 0 and row.std_s >= 0 for row in report.rows)
    # qpp-cpu is fp64: 16 bytes per amplitude.
    assert report.rows[0].est_mem_gb == pytest.approx(16 * 2**4 / 1024**3)
    assert report.meta["precision"] == {gb.CPU_TARGET: "fp64"}


def test_points_that_do_not_fit_in_memory_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pretend only 1 MiB is free: fp64 states above 15 qubits no longer fit.
    monkeypatch.setattr(hardware, "available_ram_bytes", lambda: 1024**2)
    config = gb.BenchmarkConfig(min_qubits=14, max_qubits=18, shots=10, repeats=1, targets=(gb.CPU_TARGET,))
    report = gb.run_benchmark(config)
    assert [row.n for row in report.rows] == [14, 15]
    assert [point.n for point in report.skipped] == [16, 17, 18]
    assert all("GiB" in point.reason for point in report.skipped)


def test_gpu_target_is_skipped_cleanly_without_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "gpu_target_available", lambda: False)
    config = gb.BenchmarkConfig(min_qubits=4, max_qubits=5, shots=10, repeats=1)
    report = gb.run_benchmark(config)
    assert {row.target for row in report.rows} == {gb.CPU_TARGET}
    gpu_skips = [point for point in report.skipped if point.target == gb.GPU_TARGET]
    assert len(gpu_skips) == 1 and "no NVIDIA GPU" in gpu_skips[0].reason


def test_time_budget_projection() -> None:
    rows = [
        gb.BenchmarkRow("qpp-cpu", 20, 1.0, 0.0, 0.0),
        gb.BenchmarkRow("qpp-cpu", 21, 3.0, 0.0, 0.0),
    ]
    assert gb.project_point_seconds([], 20, 6) is None
    # One point: assume the minimum 2x growth per qubit.
    assert gb.project_point_seconds(rows[:1], 22, 6) == pytest.approx(1.0 * 2**2 * 6)
    # Two points: use the observed 3x growth.
    assert gb.project_point_seconds(rows, 22, 6) == pytest.approx(3.0 * 3 * 6)


def test_time_budget_stops_the_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gb, "time_ghz", lambda n, shots, warmup, repeats: [10.0] * repeats)
    config = gb.BenchmarkConfig(
        min_qubits=4, max_qubits=7, shots=10, repeats=2, targets=(gb.CPU_TARGET,), time_budget_s=50.0
    )
    report = gb.run_benchmark(config)
    # After n=4 (10 s/run) the projection for n=5 is 10 * 2 * 3 runs = 60 s > 50 s.
    assert [row.n for row in report.rows] == [4]
    assert [point.n for point in report.skipped] == [5, 6, 7]
    assert all("time budget" in point.reason for point in report.skipped)


def test_csv_round_trip(tmp_path: Path) -> None:
    rows = [
        gb.BenchmarkRow("qpp-cpu", 20, 2.5, 0.01, 0.015625),
        gb.BenchmarkRow("nvidia", 20, 0.05, 0.001, 0.0078125),
    ]
    path = gb.write_csv(rows, tmp_path / "out" / "ghz.csv")
    assert path.read_text().splitlines()[0] == "target,n,median_s,std_s,est_mem_gb"
    assert gb.read_csv(path) == rows


def test_speedup_uses_only_sizes_measured_on_both_targets() -> None:
    rows = [
        gb.BenchmarkRow("qpp-cpu", 20, 2.0, 0.0, 0.0),
        gb.BenchmarkRow("qpp-cpu", 21, 5.0, 0.0, 0.0),
        gb.BenchmarkRow("nvidia", 21, 0.05, 0.0, 0.0),
        gb.BenchmarkRow("nvidia", 22, 0.06, 0.0, 0.0),
    ]
    assert gb.compute_speedups(rows) == [(21, pytest.approx(100.0))]


def test_plots_are_written_and_speedup_needs_gpu_data(tmp_path: Path) -> None:
    cpu_only = [gb.BenchmarkRow("qpp-cpu", n, 0.1 * 2 ** (n - 20), 0.01, 0.0) for n in (20, 21, 22)]
    assert [p.name for p in gb.make_plots(cpu_only, {}, tmp_path)] == [gb.RUNTIME_PLOT_NAME]
    both = cpu_only + [gb.BenchmarkRow("nvidia", n, 0.01, 0.001, 0.0) for n in (20, 21, 22)]
    written = gb.make_plots(both, {}, tmp_path)
    assert [p.name for p in written] == [gb.RUNTIME_PLOT_NAME, gb.SPEEDUP_PLOT_NAME]
    assert all(p.stat().st_size > 0 for p in written)


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        gb.BenchmarkConfig(min_qubits=10, max_qubits=5)
    with pytest.raises(ValueError):
        gb.BenchmarkConfig(targets=("not-a-target",))


def test_quick_flag_selects_the_reduced_sweep() -> None:
    args = gb.build_parser().parse_args(["--quick"])
    config = gb.config_from_args(args)
    assert (config.min_qubits, config.max_qubits, config.repeats, config.quick) == (16, 20, 3, True)
    assert gb.default_out_dir(quick=True).name == "quick"
    explicit = gb.config_from_args(gb.build_parser().parse_args(["--quick", "--max-qubits", "18"]))
    assert explicit.max_qubits == 18


@pytest.mark.gpu
@pytest.mark.skipif(not hardware.gpu_target_available(), reason="needs an NVIDIA GPU")
def test_ghz_on_gpu_target() -> None:
    assert gb.activate_target(gb.TARGETS[gb.GPU_TARGET]) == "fp32"
    gb.check_ghz_correctness(n=8, shots=10_000, tolerance=0.05)
    assert cudaq.get_target().name == gb.GPU_TARGET
