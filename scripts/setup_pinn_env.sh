#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DSO_DIR="${ROOT_DIR}/external/deep-symbolic-optimization-pytorch/dso"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install --upgrade wheel setuptools
"${VENV_DIR}/bin/pip" install -r "${ROOT_DIR}/requirements.txt"
"${VENV_DIR}/bin/pip" install --no-build-isolation --no-deps -e "${DSO_DIR}"

echo "PINN environment ready: ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/bin/activate"
echo "Run with: ${ROOT_DIR}/scripts/run_poisson_polynomial.sh --quick-test"
