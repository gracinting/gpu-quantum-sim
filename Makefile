# gpu-quantum-sim: one command per step, `make all` for everything.
#
#   make all      install deps, run tests, run both experiments, refresh README
#   make quick    reduced versions of both experiments (about a minute on a laptop CPU)
#   make plots    redraw every plot from the saved CSV/JSON files

PYTHON     ?= python3
VENV       ?= .venv
MAX_QUBITS ?= 30
PY         := $(VENV)/bin/python

.PHONY: all install test ghz vqe plots readme quick clean help

all: install test ghz vqe readme

help:
	@sed -n "1,6p" Makefile

$(PY):
	@$(PYTHON) -c "import sys; sys.exit(sys.version_info < (3, 11))" || \
		{ echo "Python >= 3.11 is required (CUDA-Q 0.16). Use: make PYTHON=python3.12"; exit 1; }
	$(PYTHON) -m venv $(VENV)

install: $(PY)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest

ghz:
	$(PY) src/ghz_benchmark.py --max-qubits $(MAX_QUBITS)

vqe:
	$(PY) src/h2_vqe.py

plots:
	$(PY) src/ghz_benchmark.py --plot-only
	$(PY) src/h2_vqe.py --plot-only

readme:
	$(PY) src/make_readme.py

quick:
	$(PY) src/ghz_benchmark.py --quick
	$(PY) src/h2_vqe.py --quick

clean:
	rm -rf results/quick .pytest_cache src/__pycache__ tests/__pycache__
