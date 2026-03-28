#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
OUTPUT_DIR="${ROOT_DIR}/results/poisson2d_polynomial"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/.cache/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"
mkdir -p "${OUTPUT_DIR}"

PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_case.py" \
    --case poisson2d_polynomial \
    --num-runs 20 \
    --seed 42 \
    --output-dir "${OUTPUT_DIR}" \
    --pinn-domain-points 2000 \
    --pinn-boundary-points 500 \
    --dsr-samples 1000 \
    --pinn-adam-epochs 15000 \
    --dsr-epochs 200 \
    "$@"
