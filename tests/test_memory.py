"""Tests for the statevector memory estimator and the memory guard."""

from __future__ import annotations

import pytest

import hardware


@pytest.mark.parametrize(
    ("n_qubits", "precision", "expected_bytes"),
    [
        (0, "fp32", 8),
        (1, "fp32", 16),
        (10, "fp32", 8 * 1024),
        (10, "fp64", 16 * 1024),
        (30, "fp32", 8 * 1024**3),
        (30, "fp64", 16 * 1024**3),
    ],
)
def test_estimate_matches_two_to_the_n(n_qubits: int, precision: str, expected_bytes: int) -> None:
    assert hardware.estimate_statevector_bytes(n_qubits, precision) == expected_bytes


def test_default_precision_is_single() -> None:
    assert hardware.estimate_statevector_bytes(20) == (1 << 20) * 8


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_each_extra_qubit_doubles_memory(precision: str) -> None:
    sizes = [hardware.estimate_statevector_bytes(n, precision) for n in range(20, 31)]
    assert all(later == 2 * earlier for earlier, later in zip(sizes, sizes[1:]))


def test_double_precision_costs_twice_single() -> None:
    assert hardware.estimate_statevector_bytes(25, "fp64") == 2 * hardware.estimate_statevector_bytes(25, "fp32")


def test_thirty_qubits_is_eight_gib_in_single_precision() -> None:
    assert hardware.bytes_to_gib(hardware.estimate_statevector_bytes(30, "fp32")) == pytest.approx(8.0)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        hardware.estimate_statevector_bytes(-1)
    with pytest.raises(ValueError):
        hardware.estimate_statevector_bytes(10, "fp16")


def test_fits_in_memory_applies_safety_factor() -> None:
    gib = hardware.GIB
    assert hardware.fits_in_memory(8 * gib, 16 * gib, safety_factor=0.8)
    assert not hardware.fits_in_memory(8 * gib, 9 * gib, safety_factor=0.8)
    assert hardware.fits_in_memory(8 * gib, 8 * gib, safety_factor=1.0)
    assert not hardware.fits_in_memory(1, 0)
    with pytest.raises(ValueError):
        hardware.fits_in_memory(1, 1, safety_factor=0.0)


def test_t4_fits_30_qubits_but_not_31() -> None:
    """A 16 GB T4 holds a 30-qubit fp32 state (8 GiB) but not 31 qubits (16 GiB)."""
    t4_free = 15 * hardware.GIB
    assert hardware.fits_in_memory(hardware.estimate_statevector_bytes(30, "fp32"), t4_free)
    assert not hardware.fits_in_memory(hardware.estimate_statevector_bytes(31, "fp32"), t4_free)


def test_ram_queries_are_sane() -> None:
    assert 0 < hardware.available_ram_bytes() <= hardware.total_ram_bytes()
