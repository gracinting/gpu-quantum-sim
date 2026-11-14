"""Tests for the README generator: formatting, block replacement, and sync."""

from __future__ import annotations

from pathlib import Path

import pytest

import ghz_benchmark as gb
import make_readme as mr

REPO_ROOT = Path(__file__).resolve().parent.parent


def _results(rows: list[gb.BenchmarkRow]) -> mr.Results:
    return mr.Results(ghz_rows=rows, ghz_meta={}, pes_points=[], vqe_summary=None, has_speedup_plot=False)


def test_format_helpers() -> None:
    assert mr.format_scientific(1.81e-8) == "1.8 × 10⁻⁸"
    assert mr.format_scientific(0.0) == "0"
    assert mr.format_speedup(212.4) == "212×"
    assert mr.format_speedup(3.44) == "3.4×"
    assert mr.format_duration(0.0931) == "93.1 ms"
    assert mr.format_duration(2.4221) == "2.42 s"
    assert mr.format_memory(0.25) == "256 MiB"
    assert mr.format_memory(8.0) == "8 GiB"
    assert mr.format_energy(-1.137306, 4) == "−1.1373"


def test_replace_block_swaps_only_the_named_block() -> None:
    text = "a\n<!-- AUTO:x:START -->\nold\n<!-- AUTO:x:END -->\nb\n<!-- AUTO:y:START -->\nkeep\n<!-- AUTO:y:END -->\n"
    updated = mr.replace_block(text, "x", "new \\1 text")
    assert "<!-- AUTO:x:START -->\nnew \\1 text\n<!-- AUTO:x:END -->" in updated
    assert "old" not in updated and "keep" in updated
    assert mr.replace_block(updated, "x", "new \\1 text") == updated  # idempotent


def test_replace_block_requires_exactly_one_block() -> None:
    with pytest.raises(ValueError):
        mr.replace_block("no markers here", "x", "content")


def test_headline_numbers_without_gpu_data_never_claim_a_speedup() -> None:
    results = _results([gb.BenchmarkRow("qpp-cpu", n, 1.0, 0.1, 0.25) for n in (20, 21, 22)])
    assert mr.largest_row(results).n == 22
    assert mr.headline_speedup(results) is None
    assert "not measured yet" in mr.render_summary(results)
    assert "Speedup" not in mr.render_ghz(results)


def test_headline_numbers_with_gpu_data() -> None:
    rows = [gb.BenchmarkRow("qpp-cpu", n, 10.0, 0.1, 0.25) for n in (20, 21)]
    rows += [gb.BenchmarkRow("nvidia", n, 0.05, 0.001, 0.125) for n in (20, 21, 22)]
    results = _results(rows)
    largest = mr.largest_row(results)
    assert (largest.n, largest.target) == (22, "nvidia")
    assert mr.headline_speedup(results) == (21, pytest.approx(200.0))
    summary = mr.render_summary(results)
    assert "**22 qubits**" in summary and "**200× faster**" in summary and "At 21 qubits" in summary
    table = mr.render_ghz(results)
    assert "| 21 | 2^21 |" in table and "200×" in table


def test_readme_has_every_auto_block() -> None:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for name in mr.BLOCK_NAMES:
        assert text.count(f"<!-- AUTO:{name}:START -->") == 1
        assert text.count(f"<!-- AUTO:{name}:END -->") == 1


def test_committed_readme_matches_committed_results() -> None:
    """Guards against hand-edited or stale numbers in the README."""
    results_dir = REPO_ROOT / "results"
    if not (results_dir / gb.CSV_NAME).exists() or not (results_dir / "h2_pes.csv").exists():
        pytest.skip("full results have not been generated yet")
    assert mr.main(["--check"]) == 0
