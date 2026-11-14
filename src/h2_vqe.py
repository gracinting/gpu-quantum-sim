"""VQE for molecular hydrogen with a UCCSD ansatz, validated against FCI.

Pipeline: H2 geometry -> STO-3G integrals (PySCF) -> Jordan-Wigner qubit
Hamiltonian (4 qubits, 15 Pauli terms) -> UCCSD circuit on the Hartree-Fock
reference -> classical optimizer minimising ``<psi(theta)|H|psi(theta)>``
computed with ``cudaq.observe``.

Usage::

    python src/h2_vqe.py                # full sweep, R = 0.30..2.50 A
    python src/h2_vqe.py --quick        # coarse sweep, a few seconds
    python src/h2_vqe.py --plot-only    # redraw plots from saved results
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import cudaq
import numpy as np
from scipy.optimize import minimize, minimize_scalar

import hardware
import plotting

logger = logging.getLogger("h2_vqe")

N_QUBITS = 4  # 2 spatial orbitals x 2 spins
N_ELECTRONS = 2
BASIS = "sto-3g"
EXPECTED_PAULI_TERMS = 15
CHEMICAL_ACCURACY_HA = 1.6e-3
CHEMICAL_ACCURACY_MHA = CHEMICAL_ACCURACY_HA * 1e3
EQUILIBRIUM_GUESS_ANGSTROM = 0.735
N_PARAMETERS = cudaq.kernels.uccsd_num_parameters(N_ELECTRONS, N_QUBITS)

PES_CSV_NAME = "h2_pes.csv"
CONVERGENCE_CSV_NAME = "vqe_convergence.csv"
SUMMARY_NAME = "h2_summary.json"
CONVERGENCE_PLOT_NAME = "vqe_convergence.png"
PES_PLOT_NAME = "h2_pes.png"
ERROR_PLOT_NAME = "h2_error.png"
PES_FIELDS = ("R", "E_vqe", "E_fci", "E_hf", "abs_error_mHa")

#: Errors are drawn on a log axis, which cannot show an exact zero. An error of
#: exactly zero (possible once VQE and FCI agree to the last bit) is drawn at
#: this floor, far below double-precision noise in the energy (~1e-10 mHa).
ERROR_PLOT_FLOOR_MHA = 1e-12

OPTIMIZERS = ("COBYLA", "L-BFGS-B")


@cudaq.kernel
def uccsd_ansatz(thetas: list[float], n_qubits: int, n_electrons: int):
    # Hartree-Fock reference: the lowest n_electrons spin-orbitals are occupied.
    qubits = cudaq.qvector(n_qubits)
    for i in range(n_electrons):
        x(qubits[i])
    cudaq.kernels.uccsd(qubits, thetas, n_electrons, n_qubits)


# --------------------------------------------------------------------------- #
# Hamiltonian and classical references
# --------------------------------------------------------------------------- #


def h2_geometry(bond_length: float) -> list[tuple[str, tuple[float, float, float]]]:
    """Return the H2 geometry (Angstrom) with the bond along the z axis."""
    if bond_length <= 0:
        raise ValueError(f"bond_length must be positive, got {bond_length}")
    return [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, float(bond_length)))]


@contextlib.contextmanager
def _scratch_cwd() -> Iterator[None]:
    """Run a block inside a temporary working directory.

    OpenFermion-PySCF caches integrals in an ``.hdf5`` file in the current
    directory; this keeps those files out of the repository.
    """
    original = os.getcwd()
    with tempfile.TemporaryDirectory() as scratch:
        os.chdir(scratch)
        try:
            yield
        finally:
            os.chdir(original)


def hamiltonian_from_openfermion(bond_length: float) -> cudaq.SpinOperator:
    """Build the H2 qubit Hamiltonian with PySCF + OpenFermion explicitly.

    This is the fallback (and an independent cross-check in the tests) for
    :func:`build_h2_hamiltonian`: run RHF, transform the integrals to the
    molecular-orbital basis, apply the Jordan-Wigner mapping, and convert the
    resulting ``QubitOperator`` term by term into a ``cudaq.SpinOperator``.
    """
    from cudaq import spin
    from openfermion import MolecularData, get_fermion_operator, jordan_wigner
    from openfermionpyscf import run_pyscf

    paulis = {"X": spin.x, "Y": spin.y, "Z": spin.z}
    with tempfile.TemporaryDirectory() as scratch:
        molecule = MolecularData(
            h2_geometry(bond_length),
            BASIS,
            multiplicity=1,
            charge=0,
            filename=os.path.join(scratch, "h2"),
        )
        molecule = run_pyscf(molecule, run_scf=True)
        qubit_operator = jordan_wigner(
            get_fermion_operator(molecule.get_molecular_hamiltonian())
        )

    hamiltonian: cudaq.SpinOperator | None = None
    for pauli_string, coefficient in qubit_operator.terms.items():
        if abs(coefficient.imag) > 1e-12:
            raise ValueError(f"non-Hermitian coefficient {coefficient} for {pauli_string}")
        term = spin.i(0)
        for qubit, pauli in pauli_string:
            term = term * paulis[pauli](qubit)
        term = float(coefficient.real) * term
        hamiltonian = term if hamiltonian is None else hamiltonian + term
    if hamiltonian is None:
        raise RuntimeError("Jordan-Wigner transformation produced an empty operator")
    return hamiltonian


def build_h2_hamiltonian(bond_length: float) -> cudaq.SpinOperator:
    """Return the Jordan-Wigner Hamiltonian of H2/STO-3G at ``bond_length``.

    Uses CUDA-Q's built-in chemistry helper when this CUDA-Q version has one
    and falls back to the explicit PySCF + OpenFermion route otherwise.

    Args:
        bond_length: H-H distance in Angstrom.

    Returns:
        A 4-qubit ``cudaq.SpinOperator`` with 15 Pauli terms (incl. identity).
    """
    helper = getattr(getattr(cudaq, "chemistry", None), "create_molecular_hamiltonian", None)
    if helper is None:
        logger.debug("cudaq.chemistry helper unavailable; using OpenFermion directly")
        return hamiltonian_from_openfermion(bond_length)
    with _scratch_cwd():
        hamiltonian, _ = helper(h2_geometry(bond_length), BASIS, 1, 0)
    return hamiltonian


def reference_energies(bond_length: float) -> tuple[float, float]:
    """Compute Hartree-Fock and exact FCI energies of H2/STO-3G with PySCF.

    These come straight from PySCF, independently of the qubit Hamiltonian,
    so they are a genuine reference for the VQE result.

    Returns:
        ``(E_hf, E_fci)`` in Hartree.
    """
    from pyscf import fci, gto, scf

    molecule = gto.M(
        atom=f"H 0 0 0; H 0 0 {bond_length:.10f}", basis=BASIS, unit="Angstrom", verbose=0
    )
    mean_field = scf.RHF(molecule).run()
    e_fci = fci.FCI(mean_field).kernel()[0]
    return float(mean_field.e_tot), float(e_fci)


def hydrogen_atom_energy() -> float:
    """Return the energy of one hydrogen atom in STO-3G (Hartree).

    Twice this value is the dissociation limit of the H2 curve in this basis
    (about -0.933 Ha; the complete-basis value would be exactly -1 Ha).
    """
    from pyscf import gto, scf

    atom = gto.M(atom="H 0 0 0", basis=BASIS, spin=1, verbose=0)
    return float(scf.UHF(atom).run().e_tot)


# --------------------------------------------------------------------------- #
# VQE
# --------------------------------------------------------------------------- #


@dataclass
class VQEResult:
    """Outcome of one VQE optimisation."""

    energy: float
    parameters: list[float]
    history: list[float] = field(default_factory=list)  # energy at every evaluation
    converged: bool = True

    @property
    def n_evaluations(self) -> int:
        """Number of energy evaluations (circuit executions) used."""
        return len(self.history)


@dataclass(frozen=True)
class PESPoint:
    """VQE and reference energies at one bond length. Mirrors a CSV row."""

    bond_length: float
    e_vqe: float
    e_fci: float
    e_hf: float
    abs_error_mha: float  # |E_vqe - E_fci| in milli-Hartree, at full precision
    n_evaluations: int = 0

    @classmethod
    def from_energies(
        cls, bond_length: float, e_vqe: float, e_fci: float, e_hf: float, n_evaluations: int = 0
    ) -> "PESPoint":
        """Build a point, computing the error before any rounding for storage."""
        return cls(bond_length, e_vqe, e_fci, e_hf, abs(e_vqe - e_fci) * 1e3, n_evaluations)


def energy_expectation(hamiltonian: cudaq.SpinOperator, thetas: Sequence[float]) -> float:
    """Return ``<psi(thetas)|H|psi(thetas)>`` for the UCCSD ansatz."""
    parameters = [float(theta) for theta in thetas]
    return float(
        cudaq.observe(uccsd_ansatz, hamiltonian, parameters, N_QUBITS, N_ELECTRONS).expectation()
    )


def run_vqe(
    hamiltonian: cudaq.SpinOperator,
    initial_parameters: Sequence[float] | None = None,
    optimizer: str = "COBYLA",
    max_evaluations: int = 1000,
) -> VQEResult:
    """Minimise the UCCSD energy with a SciPy optimizer.

    Args:
        hamiltonian: Qubit Hamiltonian to minimise.
        initial_parameters: Starting amplitudes; zeros (the Hartree-Fock state)
            if omitted. Pass a neighbouring geometry's optimum to warm-start.
        optimizer: ``"COBYLA"`` (gradient-free) or ``"L-BFGS-B"`` (finite
            difference gradients).
        max_evaluations: Cap on energy evaluations.

    Returns:
        The optimal energy and parameters, plus the energy at every evaluation.
    """
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {optimizer!r}")
    x0 = np.zeros(N_PARAMETERS) if initial_parameters is None else np.asarray(initial_parameters, float)
    if x0.shape != (N_PARAMETERS,):
        raise ValueError(f"expected {N_PARAMETERS} parameters, got shape {x0.shape}")

    history: list[float] = []

    def objective(thetas: np.ndarray) -> float:
        energy = energy_expectation(hamiltonian, thetas)
        history.append(energy)
        return energy

    # Tolerances were chosen by scanning the full 45-point curve: they are the
    # tightest settings at which both optimizers converge at every bond length
    # (tighter ones stall on the flat, stretched-bond landscape) and still
    # leave the energy error many orders of magnitude below chemical accuracy.
    if optimizer == "COBYLA":
        result = minimize(
            objective,
            x0,
            method="COBYLA",
            tol=1e-6,  # final trust-region radius, in radians
            options={"rhobeg": 0.1, "maxiter": max_evaluations},
        )
    else:
        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            options={"maxfun": max_evaluations, "ftol": 1e-11, "gtol": 1e-6, "eps": 1e-5},
        )
    if not result.success:
        logger.warning(
            "%s did not converge after %d evaluations: %s", optimizer, len(history), result.message
        )
    return VQEResult(
        energy=float(result.fun),
        parameters=[float(value) for value in result.x],
        history=history,
        converged=bool(result.success),
    )


def bond_length_grid(start: float = 0.30, stop: float = 2.50, step: float = 0.05) -> list[float]:
    """Return bond lengths from ``start`` to ``stop`` inclusive, free of float drift."""
    if step <= 0 or stop < start:
        raise ValueError("need step > 0 and stop >= start")
    count = int(round((stop - start) / step)) + 1
    return [round(start + index * step, 10) for index in range(count)]


def scan_potential_energy_surface(
    bond_lengths: Sequence[float], optimizer: str = "COBYLA"
) -> tuple[list[PESPoint], list[list[float]]]:
    """Run VQE along a bond-length scan, warm-starting from the previous point.

    Returns:
        The PES points and the optimal parameters at each of them.
    """
    points: list[PESPoint] = []
    optima: list[list[float]] = []
    parameters: Sequence[float] | None = None
    for bond_length in bond_lengths:
        hamiltonian = build_h2_hamiltonian(bond_length)
        result = run_vqe(hamiltonian, parameters, optimizer)
        e_hf, e_fci = reference_energies(bond_length)
        point = PESPoint.from_energies(bond_length, result.energy, e_fci, e_hf, result.n_evaluations)
        points.append(point)
        optima.append(result.parameters)
        parameters = result.parameters
        logger.info(
            "  R=%.2f A  E_vqe=%.8f  E_fci=%.8f  E_hf=%.8f  |err|=%.2e mHa  (%d evals)",
            bond_length,
            point.e_vqe,
            point.e_fci,
            point.e_hf,
            point.abs_error_mha,
            point.n_evaluations,
        )
    return points, optima


def find_equilibrium(
    points: Sequence[PESPoint],
    optima: Sequence[Sequence[float]],
    optimizer: str = "COBYLA",
    tolerance_angstrom: float = 1e-4,
) -> tuple[float, float]:
    """Locate the minimum of the VQE energy curve.

    The scan grid (0.05 A) is too coarse to pin down the bond length, so the
    VQE energy itself is minimised over R with a bounded Brent search between
    the grid neighbours of the lowest scan point.

    Returns:
        ``(R_eq in Angstrom, E_vqe(R_eq) in Hartree)``.
    """
    best = min(range(len(points)), key=lambda index: points[index].e_vqe)
    lower = points[max(best - 1, 0)].bond_length
    upper = points[min(best + 1, len(points) - 1)].bond_length
    if upper <= lower:
        return points[best].bond_length, points[best].e_vqe
    warm_start = list(optima[best])

    def vqe_energy(bond_length: float) -> float:
        return run_vqe(build_h2_hamiltonian(bond_length), warm_start, optimizer).energy

    search = minimize_scalar(
        vqe_energy, bounds=(lower, upper), method="bounded", options={"xatol": tolerance_angstrom}
    )
    return float(search.x), float(search.fun)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def write_pes_csv(points: Sequence[PESPoint], path: Path) -> Path:
    """Write the potential energy scan to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(PES_FIELDS)
        for point in points:
            writer.writerow(
                [
                    f"{point.bond_length:.2f}",
                    f"{point.e_vqe:.12f}",
                    f"{point.e_fci:.12f}",
                    f"{point.e_hf:.12f}",
                    f"{point.abs_error_mha:.3e}",
                ]
            )
    return path


def read_pes_csv(path: Path) -> list[PESPoint]:
    """Read a scan written by :func:`write_pes_csv`."""
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            PESPoint(
                bond_length=float(record["R"]),
                e_vqe=float(record["E_vqe"]),
                e_fci=float(record["E_fci"]),
                e_hf=float(record["E_hf"]),
                abs_error_mha=float(record["abs_error_mHa"]),
            )
            for record in csv.DictReader(handle)
        ]


def write_convergence_csv(history: Sequence[float], path: Path) -> Path:
    """Write the per-evaluation VQE energies to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("evaluation", "energy_ha"))
        for index, energy in enumerate(history, start=1):
            # 15 decimals keeps the full double, so the error against FCI
            # can be recomputed from this file down to the noise floor.
            writer.writerow((index, f"{energy:.15f}"))
    return path


def read_convergence_csv(path: Path) -> list[float]:
    """Read the energies written by :func:`write_convergence_csv`."""
    with path.open(newline="", encoding="utf-8") as handle:
        return [float(record["energy_ha"]) for record in csv.DictReader(handle)]


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def _minus(text: str) -> str:
    """Replace ASCII hyphens with a typographic minus sign for plot labels."""
    return text.replace("-", "\u2212")


def _shade_chemical_accuracy(ax: Any, floor: float) -> None:
    """Shade the region below 1.6 mHa on a log error axis and label it."""
    ax.axhspan(floor, CHEMICAL_ACCURACY_MHA, color=plotting.GOOD, alpha=0.10, linewidth=0, zorder=1)
    ax.axhline(CHEMICAL_ACCURACY_MHA, color=plotting.GOOD, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(
        0.99,
        CHEMICAL_ACCURACY_MHA * 0.45,
        "chemical accuracy (< 1.6 mHa)",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="top",
        fontsize=10,
        color=plotting.INK_SECONDARY,
    )


def plot_convergence(
    history: Sequence[float],
    e_fci: float,
    e_hf: float,
    bond_length: float,
    optimizer: str,
    path: Path,
) -> Path:
    """Plot the VQE energy and its error against the evaluation number."""
    plotting.apply_style()
    fig, (ax_energy, ax_error) = plotting.plt.subplots(
        2, 1, figsize=(8.0, 6.6), sharex=True, gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.12}
    )
    evaluations = np.arange(1, len(history) + 1)
    energies = np.asarray(history)

    ax_energy.plot(evaluations, energies, color=plotting.SERIES_1, zorder=3, label="VQE (UCCSD)")
    ax_energy.axhline(e_hf, color=plotting.SERIES_2, linestyle="--", linewidth=1.4, zorder=2, label="Hartree-Fock")
    ax_energy.axhline(e_fci, color=plotting.INK_SECONDARY, linewidth=1.4, zorder=2, label="FCI (exact)")
    ax_energy.set_ylabel("Energy (Ha)")
    ax_energy.legend(loc="upper right", ncols=3)
    ax_energy.grid(axis="x", visible=False)
    plotting.set_title(
        ax_energy,
        f"VQE convergence for H$_2$ at R = {bond_length:.3f} $\\AA$",
        f"UCCSD ansatz, {optimizer} from the Hartree-Fock state  |  "
        f"final E = {_minus(f'{energies[-1]:.6f}')} Ha after {len(history)} evaluations",
    )

    errors_mha = np.maximum(np.abs(energies - e_fci) * 1e3, ERROR_PLOT_FLOOR_MHA)
    floor = errors_mha.min() / 3
    ax_error.set_yscale("log")
    _shade_chemical_accuracy(ax_error, floor)
    ax_error.plot(evaluations, errors_mha, color=plotting.SERIES_1, zorder=3)
    ax_error.set_ylim(floor, max(errors_mha.max() * 3, CHEMICAL_ACCURACY_MHA * 10))
    ax_error.set_ylabel("|E$_{VQE}$ − E$_{FCI}$| (mHa)")
    ax_error.set_xlabel("Energy evaluation")
    ax_error.grid(axis="x", visible=False)
    return plotting.save_figure(fig, path)


def plot_pes(
    points: Sequence[PESPoint],
    e_dissociation: float,
    equilibrium: tuple[float, float] | None,
    path: Path,
) -> Path:
    """Plot the H2 potential energy curve: VQE points, FCI line, HF dashed."""
    plotting.apply_style()
    fig, ax = plotting.plt.subplots(figsize=(8.0, 5.2))
    bond_lengths = [point.bond_length for point in points]

    ax.axhline(e_dissociation, color=plotting.MUTED, linestyle="--", linewidth=1.2, zorder=1)
    ax.annotate(
        f"2 × E(H atom) = {_minus(f'{e_dissociation:.4f}')} Ha",
        (bond_lengths[-1], e_dissociation),
        xytext=(0, 5),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=plotting.MUTED,
    )
    ax.plot(
        bond_lengths,
        [point.e_hf for point in points],
        color=plotting.SERIES_2,
        linestyle="--",
        zorder=2,
        label="Hartree-Fock",
    )
    ax.plot(
        bond_lengths,
        [point.e_fci for point in points],
        color=plotting.INK_SECONDARY,
        zorder=3,
        label="FCI (exact)",
    )
    ax.plot(
        bond_lengths,
        [point.e_vqe for point in points],
        linestyle="none",
        marker="o",
        color=plotting.SERIES_1,
        markeredgecolor=plotting.SURFACE,
        markeredgewidth=1.2,
        zorder=4,
        label="VQE (UCCSD)",
    )
    if equilibrium is not None:
        r_eq, e_min = equilibrium
        ax.annotate(
            f"minimum: R = {r_eq:.3f} $\\AA$,  E = {_minus(f'{e_min:.4f}')} Ha",
            (r_eq, e_min),
            xytext=(22, -22),
            textcoords="offset points",
            va="center",
            fontsize=10,
            color=plotting.INK,
            arrowprops={"arrowstyle": "-", "color": plotting.MUTED, "linewidth": 1.0},
        )
    lowest = min(point.e_fci for point in points)
    highest = max(point.e_hf for point in points)
    ax.set_ylim(lowest - 0.14 * (highest - lowest), highest + 0.04 * (highest - lowest))
    ax.set_xlabel("H–H bond length R ($\\AA$)")
    ax.set_ylabel("Energy (Ha)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    max_error = max(point.abs_error_mha for point in points)
    plotting.set_title(
        ax,
        "H$_2$ potential energy curve (STO-3G)",
        f"VQE matches FCI at all {len(points)} bond lengths (max error {max_error:.1e} mHa); "
        "Hartree-Fock fails as the bond stretches",
    )
    return plotting.save_figure(fig, path)


def plot_error(points: Sequence[PESPoint], path: Path) -> Path:
    """Plot |E_vqe - E_fci| in mHa against bond length with the accuracy band."""
    plotting.apply_style()
    fig, ax = plotting.plt.subplots(figsize=(8.0, 4.8))
    bond_lengths = [point.bond_length for point in points]
    raw_errors = np.array([point.abs_error_mha for point in points])
    errors = np.maximum(raw_errors, ERROR_PLOT_FLOOR_MHA)
    floor = errors.min() / 3

    ax.set_yscale("log")
    _shade_chemical_accuracy(ax, floor)
    ax.plot(
        bond_lengths,
        errors,
        linestyle="none",
        marker="o",
        color=plotting.SERIES_1,
        markeredgecolor=plotting.SURFACE,
        markeredgewidth=1.2,
        zorder=3,
    )
    ax.set_ylim(floor, CHEMICAL_ACCURACY_MHA * 10)
    ax.set_xlabel("H–H bond length R ($\\AA$)")
    ax.set_ylabel("|E$_{VQE}$ − E$_{FCI}$| (mHa)")
    ax.grid(axis="x", visible=False)
    subtitle = f"Largest error {raw_errors.max():.1e} mHa, against a 1.6 mHa target"
    if (raw_errors < ERROR_PLOT_FLOOR_MHA).any():
        subtitle += f"  |  exact zeros drawn at {ERROR_PLOT_FLOOR_MHA:.0e} mHa"
    plotting.set_title(ax, "VQE error relative to exact FCI across the H$_2$ curve", subtitle)
    return plotting.save_figure(fig, path)


def make_plots(out_dir: Path) -> list[Path]:
    """Regenerate every VQE plot from the files saved in ``out_dir``."""
    summary = json.loads((out_dir / SUMMARY_NAME).read_text(encoding="utf-8"))
    points = read_pes_csv(out_dir / PES_CSV_NAME)
    history = read_convergence_csv(out_dir / CONVERGENCE_CSV_NAME)
    convergence = summary["convergence"]
    equilibrium = (summary["equilibrium"]["bond_length_angstrom"], summary["equilibrium"]["energy_ha"])
    return [
        plot_convergence(
            history,
            convergence["e_fci_ha"],
            convergence["e_hf_ha"],
            convergence["bond_length_angstrom"],
            summary["optimizer"],
            out_dir / CONVERGENCE_PLOT_NAME,
        ),
        plot_pes(points, summary["dissociation_limit_ha"], equilibrium, out_dir / PES_PLOT_NAME),
        plot_error(points, out_dir / ERROR_PLOT_NAME),
    ]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""
    parser = argparse.ArgumentParser(
        description="VQE (UCCSD) for H2 in STO-3G with CUDA-Q, validated against FCI."
    )
    parser.add_argument("--r-min", type=float, default=0.30, help="first bond length in A (default 0.30)")
    parser.add_argument("--r-max", type=float, default=2.50, help="last bond length in A (default 2.50)")
    parser.add_argument("--r-step", type=float, default=None, help="scan step in A (default 0.05)")
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="COBYLA", help="classical optimizer")
    parser.add_argument(
        "--target",
        choices=("qpp-cpu", "nvidia"),
        default="qpp-cpu",
        help="CUDA-Q target. The CPU default is deliberate: a 4-qubit state has "
        "16 amplitudes, far too small for a GPU to help. 'nvidia' runs in fp64.",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="output directory")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="coarse scan (0.20 A step); writes to results/quick unless --out-dir is given",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="regenerate the plots from saved results without re-running VQE",
    )
    return parser


def default_out_dir(quick: bool) -> Path:
    """Return ``results/`` (or ``results/quick/``) next to the repo root."""
    results = Path(__file__).resolve().parent.parent / "results"
    return results / "quick" if quick else results


def select_target(name: str) -> str:
    """Activate the requested target, falling back to the CPU without a GPU.

    Returns:
        The name of the target actually in use.
    """
    if name == "nvidia":
        if hardware.gpu_target_available():
            cudaq.set_target("nvidia", option="fp64")
            return "nvidia"
        logger.warning("No NVIDIA GPU available: running VQE on qpp-cpu instead of 'nvidia'.")
    cudaq.set_target("qpp-cpu")
    return "qpp-cpu"


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = args.out_dir or default_out_dir(args.quick)

    if args.plot_only:
        missing = [name for name in (SUMMARY_NAME, PES_CSV_NAME, CONVERGENCE_CSV_NAME) if not (out_dir / name).exists()]
        if missing:
            logger.error("Nothing to plot: %s missing in %s. Run the VQE first.", missing, out_dir)
            return 1
        for path in make_plots(out_dir):
            logger.info("Wrote %s", path)
        return 0

    target = select_target(args.target)
    step = args.r_step if args.r_step is not None else (0.20 if args.quick else 0.05)
    bond_lengths = bond_length_grid(args.r_min, args.r_max, step)

    # 1. Single-point VQE at the experimental-like geometry, cold-started from HF.
    r0 = EQUILIBRIUM_GUESS_ANGSTROM
    hamiltonian = build_h2_hamiltonian(r0)
    n_terms = hamiltonian.term_count
    logger.info(
        "H2/%s at R=%.3f A: %d qubits, %d Pauli terms (incl. identity), %d UCCSD parameters, target %s",
        BASIS, r0, hamiltonian.qubit_count, n_terms, N_PARAMETERS, target,
    )
    if n_terms != EXPECTED_PAULI_TERMS:
        logger.warning("Expected %d Pauli terms, got %d", EXPECTED_PAULI_TERMS, n_terms)
    single = run_vqe(hamiltonian, optimizer=args.optimizer)
    e_hf0, e_fci0 = reference_energies(r0)
    single_error_mha = abs(single.energy - e_fci0) * 1e3
    logger.info(
        "VQE at R=%.3f A: E=%.8f Ha, FCI=%.8f Ha, |error|=%.2e mHa, %d evaluations",
        r0, single.energy, e_fci0, single_error_mha, single.n_evaluations,
    )

    # 2. Potential energy scan, warm-started point to point.
    logger.info("Scanning %d bond lengths from %.2f to %.2f A", len(bond_lengths), bond_lengths[0], bond_lengths[-1])
    points, optima = scan_potential_energy_surface(bond_lengths, args.optimizer)

    # 3. Equilibrium geometry from the VQE energy itself.
    r_eq, e_min = find_equilibrium(points, optima, args.optimizer)
    _, e_fci_eq = reference_energies(r_eq)
    max_error_point = max(points, key=lambda point: point.abs_error_mha)
    e_dissociation = 2.0 * hydrogen_atom_energy()

    summary = {
        "molecule": "H2",
        "basis": BASIS,
        "mapping": "Jordan-Wigner",
        "n_qubits": N_QUBITS,
        "n_pauli_terms": n_terms,
        "n_uccsd_parameters": N_PARAMETERS,
        "optimizer": args.optimizer,
        "target": target,
        "quick": bool(args.quick),
        "convergence": {
            "bond_length_angstrom": r0,
            "e_vqe_ha": single.energy,
            "e_fci_ha": e_fci0,
            "e_hf_ha": e_hf0,
            "abs_error_mha": single_error_mha,
            "n_evaluations": single.n_evaluations,
        },
        "scan": {
            "n_points": len(points),
            "r_min_angstrom": bond_lengths[0],
            "r_max_angstrom": bond_lengths[-1],
            "step_angstrom": step,
            "max_abs_error_mha": max_error_point.abs_error_mha,
            "max_abs_error_at_angstrom": max_error_point.bond_length,
            "total_evaluations": sum(point.n_evaluations for point in points),
        },
        "equilibrium": {
            "bond_length_angstrom": r_eq,
            "energy_ha": e_min,
            "e_fci_ha": e_fci_eq,
        },
        "dissociation_limit_ha": e_dissociation,
        "chemical_accuracy_mha": CHEMICAL_ACCURACY_MHA,
        "hardware": hardware.describe_hardware(),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_pes_csv(points, out_dir / PES_CSV_NAME)
    write_convergence_csv(single.history, out_dir / CONVERGENCE_CSV_NAME)
    (out_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for name in (PES_CSV_NAME, CONVERGENCE_CSV_NAME, SUMMARY_NAME):
        logger.info("Wrote %s", out_dir / name)
    for path in make_plots(out_dir):
        logger.info("Wrote %s", path)

    logger.info("Equilibrium bond length (VQE): %.4f A", r_eq)
    logger.info("Minimum energy (VQE):          %.6f Ha  (FCI at the same R: %.6f Ha)", e_min, e_fci_eq)
    logger.info(
        "Max |E_vqe - E_fci| over the scan: %.2e mHa at R=%.2f A  (chemical accuracy: %.1f mHa)",
        max_error_point.abs_error_mha, max_error_point.bond_length, CHEMICAL_ACCURACY_MHA,
    )
    cudaq.reset_target()
    if max_error_point.abs_error_mha > CHEMICAL_ACCURACY_MHA:
        logger.error("VQE did NOT reach chemical accuracy at every bond length.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
