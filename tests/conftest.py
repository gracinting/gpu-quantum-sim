"""Shared pytest fixtures."""

from __future__ import annotations

import cudaq
import pytest


@pytest.fixture(autouse=True)
def cpu_target():
    """Run every test on the CPU statevector target unless it opts out.

    The active target is process-global state in CUDA-Q, so it is reset after
    each test to keep tests independent of execution order.
    """
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()
