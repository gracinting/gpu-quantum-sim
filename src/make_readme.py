"""Fill the auto-generated blocks of README.md from the files in ``results/``.

Every number in the README's marked blocks is computed here from the CSV and
JSON files that the experiments wrote, so the prose can never drift from the
data. Blocks are delimited by HTML comments::

    <!-- AUTO:summary:START -->
    ...generated, do not edit by hand...
    <!-- AUTO:summary:END -->

Usage::

    python src/make_readme.py            # rewrite README.md in place
    python src/make_readme.py --check    # exit 1 if README.md is out of date
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import ghz_benchmark as gb
import h2_vqe

logger = logging.getLogger("make_readme")

REPO_ROOT = Path(__file__).resolve().parent.parent
BLOCK_NAMES = ("summary", "ghz", "vqe", "hardware", "resume")
_SUPERSCRIPTS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


@dataclass
class Results:
    """Everything the experiments left on disk. Missing parts are empty/None."""

    ghz_rows: list[gb.BenchmarkRow]
    ghz_meta: dict[str, Any]
    pes_points: list[h2_vqe.PESPoint]
    vqe_summary: dict[str, Any] | None
    has_speedup_plot: bool


def load_results(results_dir: Path) -> Results:
    """Read whichever result files exist in ``results_dir``."""
    ghz_csv = results_dir / gb.CSV_NAME
    ghz_meta = results_dir / gb.META_NAME
    pes_csv = results_dir / h2_vqe.PES_CSV_NAME
    summary = results_dir / h2_vqe.SUMMARY_NAME
    return Results(
        ghz_rows=gb.read_csv(ghz_csv) if ghz_csv.exists() else [],
        ghz_meta=json.loads(ghz_meta.read_text(encoding="utf-8")) if ghz_meta.exists() else {},
        pes_points=h2_vqe.read_pes_csv(pes_csv) if pes_csv.exists() else [],
        vqe_summary=json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else None,
        has_speedup_plot=(results_dir / gb.SPEEDUP_PLOT_NAME).exists(),
    )


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #


def format_scientific(value: float, digits: int = 1) -> str:
    """Format ``value`` as ``a × 10ⁿ`` for prose, e.g. ``1.8 × 10⁻⁸``."""
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"{mantissa} × 10{str(int(exponent)).translate(_SUPERSCRIPTS)}"


def format_speedup(speedup: float) -> str:
    """Format a speedup factor, e.g. ``212×`` or ``3.4×``."""
    return f"{speedup:.0f}×" if speedup >= 10 else f"{speedup:.1f}×"


def format_duration(seconds: float) -> str:
    """Format a duration for a table cell."""
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def format_memory(gib: float) -> str:
    """Format a statevector size for a table cell."""
    if gib < 1.0:
        return f"{gib * 1024:.0f} MiB"
    return f"{gib:.0f} GiB" if float(gib).is_integer() else f"{gib:.1f} GiB"


def format_energy(hartree: float, digits: int = 6) -> str:
    """Format an energy with a typographic minus sign, e.g. ``−1.1373``."""
    return f"{hartree:.{digits}f}".replace("-", "−")


def max_vqe_error_mha(results: Results) -> float | None:
    """Largest |E_vqe - E_fci| over the scan in mHa, from ``h2_pes.csv``."""
    if not results.pes_points:
        return None
    return max(point.abs_error_mha for point in results.pes_points)


# --------------------------------------------------------------------------- #
# Derived headline numbers
# --------------------------------------------------------------------------- #


def largest_row(results: Results) -> gb.BenchmarkRow | None:
    """Return the row with the most qubits (GPU wins ties: it is the headline)."""
    if not results.ghz_rows:
        return None
    return max(results.ghz_rows, key=lambda row: (row.n, row.target == gb.GPU_TARGET))


def headline_speedup(results: Results) -> tuple[int, float] | None:
    """Return ``(n, speedup)`` at the largest n measured on both targets."""
    speedups = gb.compute_speedups(results.ghz_rows)
    return speedups[-1] if speedups else None


def _device_name(results: Results, target: str) -> str:
    hw = results.ghz_meta.get("hardware", {})
    if target == gb.GPU_TARGET:
        return (hw.get("gpu") or {}).get("name", "an NVIDIA GPU")
    return hw.get("cpu", "the CPU")


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #


def render_summary(results: Results) -> str:
    """One-paragraph summary with the headline numbers."""
    sentences = [
        "**gpu-quantum-sim** simulates quantum circuits on classical hardware with "
        "[NVIDIA CUDA-Q](https://developer.nvidia.com/cuda-q) and uses that simulator to "
        "solve a small quantum-chemistry problem end to end."
    ]
    largest = largest_row(results)
    speedup = headline_speedup(results)
    if largest is not None:
        backend = "GPU (cuStateVec)" if largest.target == gb.GPU_TARGET else "CPU backend"
        sentences.append(
            f"GHZ circuits were simulated up to **{largest.n} qubits** "
            f"({format_memory(largest.est_mem_gb)} statevector) on the {backend} "
            f"({_device_name(results, largest.target)})."
        )
    if speedup is not None:
        n, factor = speedup
        sentences.append(
            f"At {n} qubits, the largest size both backends completed, the GPU was "
            f"**{format_speedup(factor)} faster** than CUDA-Q's CPU backend."
        )
    elif largest is not None:
        sentences.append(
            "**GPU speedup: not measured yet.** The committed results come from a machine "
            "without an NVIDIA GPU, so the GPU half of the benchmark was detected as "
            "unavailable and skipped; "
            "[run the Colab notebook](notebooks/colab_quickstart.ipynb) on a free T4 to "
            "produce it."
        )
    error = max_vqe_error_mha(results)
    if error is not None and results.vqe_summary:
        scan = results.vqe_summary["scan"]
        eq = results.vqe_summary["equilibrium"]
        sentences.append(
            f"A VQE with a UCCSD ansatz reproduces the H₂ potential energy curve at "
            f"{scan['n_points']} bond lengths with a maximum error of "
            f"**{format_scientific(error)} mHa** against exact FCI (chemical accuracy is "
            f"1.6 mHa) and finds the equilibrium bond length at "
            f"**{eq['bond_length_angstrom']:.3f} Å** with "
            f"E = **{format_energy(eq['energy_ha'], 4)} Ha**."
        )
    sentences.append(
        "Every number on this page is generated from the CSV files in "
        "[`results/`](results) by [`src/make_readme.py`](src/make_readme.py)."
    )
    return " ".join(sentences)


def render_ghz(results: Results) -> str:
    """Benchmark table, the speedup plot when it exists, and skip notes."""
    if not results.ghz_rows:
        return "_No benchmark results yet. Run `make ghz`._"
    cpu = {row.n: row for row in gb.rows_for(results.ghz_rows, gb.CPU_TARGET)}
    gpu = {row.n: row for row in gb.rows_for(results.ghz_rows, gb.GPU_TARGET)}
    precision = results.ghz_meta.get("precision", {})
    cpu_precision = precision.get(gb.CPU_TARGET, "fp64")
    gpu_precision = precision.get(gb.GPU_TARGET, "fp32")
    lines: list[str] = []

    if results.has_speedup_plot and gpu:
        lines += ["![GPU speedup over CPU against qubit count](results/ghz_speedup.png)", ""]

    def cell(row: gb.BenchmarkRow | None) -> str:
        if row is None:
            return "—"
        return f"{format_duration(row.median_s)} ± {format_duration(row.std_s)}"

    if gpu:
        lines += [
            f"| Qubits | Amplitudes | CPU `qpp-cpu` ({cpu_precision}) | "
            f"GPU `nvidia` ({gpu_precision}) | Speedup |",
            "|---:|---:|---:|---:|---:|",
        ]
        for n in sorted(cpu.keys() | gpu.keys()):
            ratio = (
                format_speedup(cpu[n].median_s / gpu[n].median_s) if n in cpu and n in gpu else "—"
            )
            lines.append(f"| {n} | 2^{n} | {cell(cpu.get(n))} | {cell(gpu.get(n))} | {ratio} |")
    else:
        lines += [
            f"| Qubits | Amplitudes | Statevector ({cpu_precision}) | CPU `qpp-cpu` median ± std |",
            "|---:|---:|---:|---:|",
        ]
        for n in sorted(cpu):
            lines.append(
                f"| {n} | 2^{n} | {format_memory(cpu[n].est_mem_gb)} | {cell(cpu[n])} |"
            )

    config = results.ghz_meta.get("config", {})
    if config:
        lines += [
            "",
            f"Each cell is the median ± standard deviation of {config['repeats']} timed runs "
            f"after {config['warmup']} warm-up, {config['shots']} shots per run, timing the "
            "whole `cudaq.sample` call (state allocation, gates and sampling).",
        ]

    notes = _skip_notes(results)
    if notes:
        lines += ["", "Points that were deliberately not run:", ""] + notes
    return "\n".join(lines)


def _skip_kind(reason: str) -> str:
    """Classify a skip reason so consecutive points can be grouped."""
    if reason.startswith("statevector needs"):
        return "memory"
    if reason.startswith("projected"):
        return "time"
    return "other"


def _skip_notes(results: Results) -> list[str]:
    """Summarise skipped points: one bullet per target and kind of reason."""
    skipped = results.ghz_meta.get("skipped", [])
    notes: list[str] = []
    for target in (gb.CPU_TARGET, gb.GPU_TARGET):
        target_skips = [point for point in skipped if point["target"] == target]
        whole_target = [point for point in target_skips if point["n"] is None]
        if whole_target:
            notes.append(f"- `{target}`: {whole_target[0]['reason']}")
            continue
        groups: dict[str, list[dict[str, Any]]] = {}
        for point in target_skips:
            groups.setdefault(_skip_kind(point["reason"]), []).append(point)
        for points in groups.values():
            sizes = [point["n"] for point in points]
            span = f"n = {sizes[0]}" if len(sizes) == 1 else f"n = {sizes[0]}–{sizes[-1]}"
            lead = "" if len(sizes) == 1 else f"at n = {sizes[0]}, "
            notes.append(f"- `{target}`, {span}: {lead}{points[0]['reason']}.")
    return notes


def render_vqe(results: Results) -> str:
    """Key VQE numbers as a table."""
    summary = results.vqe_summary
    if not summary or not results.pes_points:
        return "_No VQE results yet. Run `make vqe`._"
    conv = summary["convergence"]
    scan = summary["scan"]
    eq = summary["equilibrium"]
    worst = max(results.pes_points, key=lambda point: point.abs_error_mha)
    worst_hf = max(results.pes_points, key=lambda point: point.e_hf - point.e_fci)
    hf_error_mha = (worst_hf.e_hf - worst_hf.e_fci) * 1e3
    rows = [
        ("Qubits / Pauli terms in the Hamiltonian", f"{summary['n_qubits']} / {summary['n_pauli_terms']} (including the identity)"),
        ("UCCSD parameters", f"{summary['n_uccsd_parameters']} (2 single + 1 double excitation)"),
        (
            f"VQE energy at R = {conv['bond_length_angstrom']:.3f} Å",
            f"{format_energy(conv['e_vqe_ha'])} Ha after {conv['n_evaluations']} energy evaluations "
            f"({summary['optimizer']}, cold start from Hartree-Fock)",
        ),
        (f"FCI energy at R = {conv['bond_length_angstrom']:.3f} Å", f"{format_energy(conv['e_fci_ha'])} Ha"),
        ("Error at that geometry", f"{format_scientific(conv['abs_error_mha'])} mHa"),
        (
            "Bond-length scan",
            f"{scan['n_points']} points, {scan['r_min_angstrom']:.2f}–{scan['r_max_angstrom']:.2f} Å "
            f"in {scan['step_angstrom']:.2f} Å steps, warm-started point to point",
        ),
        (
            "**Maximum VQE error over the scan**",
            f"**{format_scientific(worst.abs_error_mha)} mHa** at R = "
            f"{worst.bond_length:.2f} Å (target: < 1.6 mHa)",
        ),
        (
            "Maximum Hartree-Fock error over the scan",
            f"{hf_error_mha:.0f} mHa at R = {worst_hf.bond_length:.2f} Å",
        ),
        ("**Equilibrium bond length (VQE)**", f"**{eq['bond_length_angstrom']:.3f} Å**"),
        ("**Minimum energy (VQE)**", f"**{format_energy(eq['energy_ha'], 4)} Ha**"),
    ]
    lines = ["| Quantity | Result |", "|:---|:---|"] + [f"| {name} | {value} |" for name, value in rows]
    return "\n".join(lines)


def render_hardware(results: Results) -> str:
    """Describe the machine that produced the committed results."""
    hw = results.ghz_meta.get("hardware") or (results.vqe_summary or {}).get("hardware")
    if not hw:
        return "_No results yet._"
    gpu = hw.get("gpu")
    gpu_text = (
        f"{gpu['name']} ({gpu['total_bytes'] / 1024**3:.0f} GiB VRAM)"
        if gpu
        else "none detected, so the `nvidia` target was skipped at runtime"
    )
    lines = [
        f"- **CPU:** {hw['cpu']} ({hw['cpu_cores']} cores), {hw['ram_gib']:.0f} GiB RAM",
        f"- **GPU:** {gpu_text}",
        f"- **OS:** {hw['os']}",
        f"- **Software:** CUDA-Q {hw['cudaq']}, Python {hw['python']}",
    ]
    generated = results.ghz_meta.get("generated_utc")
    if generated:
        lines.append(f"- **Benchmark run:** {generated[:10]} (UTC)")
    return "\n".join(lines)


def render_resume(results: Results) -> str:
    """A résumé bullet built only from measured numbers."""
    largest = largest_row(results)
    speedup = headline_speedup(results)
    error = max_vqe_error_mha(results)
    if largest is None or error is None:
        return "_Run both experiments to generate this bullet._"
    vqe_clause = (
        "VQE with a UCCSD ansatz reproduces the H₂ potential energy curve within "
        f"{format_scientific(error)} mHa of FCI (chemical accuracy)."
    )
    if speedup is not None and largest.target == gb.GPU_TARGET:
        bullet = (
            f"GPU-accelerated statevector simulation up to {largest.n} qubits with NVIDIA "
            f"CUDA-Q/cuStateVec ({format_speedup(speedup[1])} speedup over CPU at "
            f"{speedup[0]} qubits); {vqe_clause}"
        )
        return f"> {bullet}"
    bullet = (
        f"Statevector simulation of GHZ circuits up to {largest.n} qubits with NVIDIA CUDA-Q "
        f"(CPU backend), with a GPU/cuStateVec benchmark path that detects hardware at "
        f"runtime; {vqe_clause}"
    )
    return (
        f"> {bullet}\n\n"
        "This is the CPU-only version, because no GPU speedup has been measured yet. After "
        "a GPU run, `make readme` replaces it with the form "
        "“GPU-accelerated statevector simulation up to N qubits with NVIDIA "
        "CUDA-Q/cuStateVec (X× speedup over CPU); …” using the measured N and X."
    )


RENDERERS = {
    "summary": render_summary,
    "ghz": render_ghz,
    "vqe": render_vqe,
    "hardware": render_hardware,
    "resume": render_resume,
}


# --------------------------------------------------------------------------- #
# Block replacement
# --------------------------------------------------------------------------- #


def replace_block(text: str, name: str, content: str) -> str:
    """Replace the body of the ``AUTO:name`` block in ``text``.

    Raises:
        ValueError: If the block markers are missing or duplicated.
    """
    start, end = f"<!-- AUTO:{name}:START -->", f"<!-- AUTO:{name}:END -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if len(pattern.findall(text)) != 1:
        raise ValueError(f"README must contain exactly one AUTO:{name} block")
    return pattern.sub(lambda _: f"{start}\n{content}\n{end}", text)


def render_readme(template: str, results: Results) -> str:
    """Return ``template`` with every auto block regenerated from ``results``."""
    text = template
    for name in BLOCK_NAMES:
        text = replace_block(text, name, RENDERERS[name](results))
    return text


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Regenerate the data-driven blocks of README.md.")
    parser.add_argument("--readme", type=Path, default=REPO_ROOT / "README.md")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--check", action="store_true", help="exit 1 if README.md is out of date")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    current = args.readme.read_text(encoding="utf-8")
    updated = render_readme(current, load_results(args.results_dir))
    if args.check:
        if current != updated:
            logger.error("%s is out of date; run `python src/make_readme.py`.", args.readme)
            return 1
        logger.info("%s is up to date.", args.readme)
        return 0
    if current == updated:
        logger.info("%s already up to date.", args.readme)
    else:
        args.readme.write_text(updated, encoding="utf-8")
        logger.info("Updated %s", args.readme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
