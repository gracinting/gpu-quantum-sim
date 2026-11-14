"""Tests for the H2 Hamiltonian, the UCCSD ansatz and the VQE loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import h2_vqe

R_EQ = 0.735


@pytest.fixture(scope="module")
def hamiltonian():
    return h2_vqe.build_h2_hamiltonian(R_EQ)


@pytest.fixture(scope="module")
def references() -> tuple[float, float]:
    return h2_vqe.reference_energies(R_EQ)


def test_hamiltonian_has_15_pauli_terms_on_4_qubits(hamiltonian) -> None:
    assert hamiltonian.term_count == h2_vqe.EXPECTED_PAULI_TERMS == 15
    assert hamiltonian.qubit_count == 4


@pytest.mark.parametrize("bond_length", [0.30, 1.50, 2.50])
def test_term_count_is_15_along_the_whole_curve(bond_length: float) -> None:
    assert h2_vqe.build_h2_hamiltonian(bond_length).term_count == 15


def test_builtin_helper_matches_explicit_openfermion_route(hamiltonian) -> None:
    explicit = h2_vqe.hamiltonian_from_openfermion(R_EQ)
    assert explicit.term_count == 15
    np.testing.assert_allclose(
        np.array(explicit.to_matrix()), np.array(hamiltonian.to_matrix()), atol=1e-12
    )


def test_building_the_hamiltonian_leaves_no_files_behind(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    h2_vqe.build_h2_hamiltonian(0.9)
    h2_vqe.hamiltonian_from_openfermion(0.9)
    assert list(tmp_path.iterdir()) == []


def test_ground_state_of_qubit_hamiltonian_is_fci(hamiltonian, references) -> None:
    _, e_fci = references
    ground = np.linalg.eigvalsh(np.array(hamiltonian.to_matrix()))[0]
    assert ground == pytest.approx(e_fci, abs=1e-9)


def test_reference_energies_at_equilibrium(references) -> None:
    e_hf, e_fci = references
    assert e_hf == pytest.approx(-1.116998996754, abs=1e-8)
    assert e_fci == pytest.approx(-1.137306035753, abs=1e-8)
    assert e_fci < e_hf


def test_uccsd_needs_three_parameters() -> None:
    assert h2_vqe.N_PARAMETERS == 3  # two singles + one double excitation


def test_ansatz_at_zero_is_the_hartree_fock_state(hamiltonian, references) -> None:
    e_hf, _ = references
    energy = h2_vqe.energy_expectation(hamiltonian, [0.0] * h2_vqe.N_PARAMETERS)
    assert energy == pytest.approx(e_hf, abs=1e-9)


@pytest.mark.parametrize("optimizer", h2_vqe.OPTIMIZERS)
def test_vqe_reaches_chemical_accuracy_at_equilibrium(hamiltonian, references, optimizer: str) -> None:
    _, e_fci = references
    result = h2_vqe.run_vqe(hamiltonian, optimizer=optimizer)
    error_mha = abs(result.energy - e_fci) * 1e3
    assert error_mha < h2_vqe.CHEMICAL_ACCURACY_MHA == 1.6
    assert result.energy == pytest.approx(-1.1373, abs=1e-4)
    # The variational principle: no trial state can go below the exact energy.
    assert result.energy >= e_fci - 1e-9
    assert result.history[0] > result.energy
    assert result.n_evaluations == len(result.history) > 1


def test_uccsd_is_numerically_exact_for_h2(hamiltonian, references) -> None:
    """Two electrons: singles + doubles span the full space, so UCCSD == FCI."""
    _, e_fci = references
    result = h2_vqe.run_vqe(hamiltonian, optimizer="COBYLA")
    assert abs(result.energy - e_fci) * 1e3 < 1e-5  # mHa


def test_run_vqe_validates_inputs(hamiltonian) -> None:
    with pytest.raises(ValueError):
        h2_vqe.run_vqe(hamiltonian, optimizer="SGD")
    with pytest.raises(ValueError):
        h2_vqe.run_vqe(hamiltonian, initial_parameters=[0.0])


def test_bond_length_grid_matches_the_spec() -> None:
    grid = h2_vqe.bond_length_grid(0.30, 2.50, 0.05)
    assert len(grid) == 45
    assert grid[0] == 0.30 and grid[-1] == 2.50
    assert grid[9] == 0.75
    assert np.allclose(np.diff(grid), 0.05)


def test_warm_started_scan_stays_within_chemical_accuracy() -> None:
    bond_lengths = [0.5, 0.75, 1.5, 2.5]
    points, optima = h2_vqe.scan_potential_energy_surface(bond_lengths)
    assert [point.bond_length for point in points] == bond_lengths
    assert len(optima) == len(points)
    for point in points:
        assert point.abs_error_mha < h2_vqe.CHEMICAL_ACCURACY_MHA
        assert point.e_fci <= point.e_hf
    # Hartree-Fock gets worse as the bond stretches (static correlation).
    assert points[-1].e_hf - points[-1].e_fci > 10 * (points[1].e_hf - points[1].e_fci)


def test_equilibrium_search_finds_the_known_minimum() -> None:
    points, optima = h2_vqe.scan_potential_energy_surface([0.65, 0.70, 0.75, 0.80])
    r_eq, e_min = h2_vqe.find_equilibrium(points, optima)
    assert r_eq == pytest.approx(0.735, abs=0.005)
    assert e_min == pytest.approx(-1.1373, abs=1e-4)
    assert e_min <= min(point.e_vqe for point in points) + 1e-9


def test_dissociation_limit_is_two_hydrogen_atoms() -> None:
    e_h = h2_vqe.hydrogen_atom_energy()
    assert e_h == pytest.approx(-0.46658185, abs=1e-6)
    _, e_fci_far = h2_vqe.reference_energies(6.0)
    assert e_fci_far == pytest.approx(2 * e_h, abs=1e-4)


def test_pes_csv_round_trip(tmp_path: Path) -> None:
    points = [h2_vqe.PESPoint.from_energies(0.75, -1.137117000012, -1.137117, -1.116151, 40)]
    path = h2_vqe.write_pes_csv(points, tmp_path / "pes.csv")
    assert path.read_text().splitlines()[0] == "R,E_vqe,E_fci,E_hf,abs_error_mHa"
    loaded = h2_vqe.read_pes_csv(path)
    assert loaded[0].bond_length == 0.75
    assert loaded[0].e_vqe == pytest.approx(-1.137117)
    assert loaded[0].e_hf == pytest.approx(-1.116151)
    # The error column survives the round trip even though it is far below
    # what the 12-decimal energy columns alone could resolve to 3 digits.
    assert loaded[0].abs_error_mha == pytest.approx(1.2e-8, rel=1e-2)


def test_quick_cli_writes_every_artifact(tmp_path: Path) -> None:
    exit_code = h2_vqe.main(["--quick", "--r-step", "0.55", "--out-dir", str(tmp_path)])
    assert exit_code == 0
    expected = {
        h2_vqe.PES_CSV_NAME,
        h2_vqe.CONVERGENCE_CSV_NAME,
        h2_vqe.SUMMARY_NAME,
        h2_vqe.CONVERGENCE_PLOT_NAME,
        h2_vqe.PES_PLOT_NAME,
        h2_vqe.ERROR_PLOT_NAME,
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    assert h2_vqe.main(["--plot-only", "--out-dir", str(tmp_path)]) == 0
