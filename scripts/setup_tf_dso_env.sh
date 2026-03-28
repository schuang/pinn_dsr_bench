#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv_tf"
TF_REQS="${ROOT_DIR}/requirements-tf.txt"
DSO_DIR="${ROOT_DIR}/external/deep-symbolic-optimization/dso"
CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    if [[ -n "${CONDA_BIN}" ]]; then
        "${CONDA_BIN}" create -y -p "${VENV_DIR}" python=3.7 pip
    else
        echo "Missing ${VENV_DIR}. Install conda/miniforge or create a Python 3.7 env at ${VENV_DIR} first." >&2
        exit 1
    fi
fi

"${VENV_DIR}/bin/pip" install --upgrade "pip<24.1" "setuptools<69" wheel==0.42.0
"${VENV_DIR}/bin/pip" install --use-deprecated=legacy-resolver -r "${TF_REQS}"
"${VENV_DIR}/bin/pip" install --use-deprecated=legacy-resolver --no-build-isolation --no-deps -e "${DSO_DIR}"

echo "TensorFlow DSO environment ready: ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/bin/activate"
echo "Use from benchmark scripts with: --dsr-backend tensorflow"
