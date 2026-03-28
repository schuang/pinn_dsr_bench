# PINN DSR Bench

`pinn_dsr_bench` is a standalone benchmark for symbolic regression from PINN
solutions on two built-in PDE cases:

- `poisson2d_polynomial`: `x1**4 + 1.2*x2**4`
- `advection2d_gaussian`: `exp(-((x1 - t)**2 + (x2 - t)**2) / 0.5)`

The repository contains the PINN training code, PDE definitions, benchmark
runner, and DSR integration needed to run these cases end to end.

## Included Cases

### Poisson 2D Polynomial

- Elliptic PDE on `[0, 1] x [0, 1]`
- Ground-truth solution: `x1**4 + 1.2*x2**4`
- Symbol library: `add, sub, mul, div, sin, cos, exp, log, const, n2, n4`
- Default script: `./scripts/run_poisson_polynomial.sh`

### Advection 2D Gaussian

- Time-dependent advection PDE
- Ground-truth solution: `exp(-((x1 - t)**2 + (x2 - t)**2) / 0.5)`
- Symbol library: `add, sub, mul, div, sin, cos, exp, log, const`
- Default script: `./scripts/run_advection_gaussian.sh`

## Layout

```text
pinn_dsr_bench/
├── README.md
├── requirements.txt
├── scripts/
│   ├── run_case.py
│   ├── run_poisson_polynomial.sh
│   ├── run_advection_gaussian.sh
│   ├── setup_pinn_env.sh
│   └── setup_tf_dso_env.sh
└── src/pinn_dsr_bench/
    ├── benchmark/
    ├── dsr/
    ├── pde/
    └── pinn/
```

## Setup

The main environment lives in `.venv`:

```bash
cd pinn_dsr_bench
./scripts/setup_pinn_env.sh
source .venv/bin/activate
```

The default backend installs the vendored PyTorch DSO package from:

```text
external/deep-symbolic-optimization-pytorch/dso
```

`./scripts/setup_pinn_env.sh` is equivalent to:

```bash
cd pinn_dsr_bench
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install --upgrade wheel setuptools
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --no-build-isolation --no-deps -e external/deep-symbolic-optimization-pytorch/dso
```

Those commands are enough to reproduce the `.venv` setup manually.

## TensorFlow Backend

An optional TensorFlow DSO environment can be created in `.venv_tf`:

```bash
cd pinn_dsr_bench
./scripts/setup_tf_dso_env.sh
source .venv_tf/bin/activate
```

The TensorFlow backend installs the vendored TensorFlow DSO package from:

```text
external/deep-symbolic-optimization/dso
```

If `conda` is available and `.venv_tf` does not already exist, the setup script
creates a Python 3.7 environment first. The full setup is equivalent to:

```bash
cd pinn_dsr_bench
conda create -y -p .venv_tf python=3.7 pip
.venv_tf/bin/pip install --upgrade "pip<24.1" "setuptools<69" wheel==0.42.0
.venv_tf/bin/pip install --use-deprecated=legacy-resolver -r requirements-tf.txt
.venv_tf/bin/pip install --use-deprecated=legacy-resolver --no-build-isolation --no-deps -e external/deep-symbolic-optimization/dso
```

If `.venv_tf` already exists with a usable Python 3.7 interpreter, reuse it and
run the `pip install` commands only.

This backend uses pinned legacy versions:

- Python `3.7`
- TensorFlow `1.15.5`
- NumPy `1.19.5`
- Pandas `1.1.5`
- SciPy `1.5.4`

## Running Benchmarks

The shell wrappers use `pinn_dsr_bench/.venv/bin/python` by default, so they
run inside the local benchmark environment even if your shell is not activated.

PyTorch DSO now defaults to CPU execution. If you want to use a GPU explicitly,
add `--dsr-device cuda`.

Run the two built-in cases:

```bash
./scripts/run_poisson_polynomial.sh
./scripts/run_advection_gaussian.sh
```

Use the TensorFlow backend when needed:

```bash
./scripts/run_poisson_polynomial.sh --dsr-backend tensorflow
./scripts/run_advection_gaussian.sh --dsr-backend tensorflow
```

List all supported cases:

```bash
python3 ./scripts/run_case.py --list-cases
```

## Default Configurations

### Poisson 2D Polynomial

- `20` runs with base seed `42`
- PINN: `4x50 tanh`, `2000` domain points, `500` boundary points,
  `15000` Adam epochs, `L-BFGS` enabled
- DSR: `1000` PINN samples, `200` epochs

### Advection 2D Gaussian

- `1` run with base seed `42`
- PINN: `4x50 tanh`, `2000` domain points, `500` boundary points,
  `15000` Adam epochs, `L-BFGS` enabled
- DSR: `1000` PINN samples, `200` epochs

## Useful Flags

All wrapper scripts forward extra CLI arguments to `scripts/run_case.py`.

```bash
./scripts/run_poisson_polynomial.sh --quick-test
./scripts/run_poisson_polynomial.sh --num-runs 20
./scripts/run_advection_gaussian.sh --output-dir results/custom_advection
./scripts/run_poisson_polynomial.sh --dsr-timeout 7200
./scripts/run_advection_gaussian.sh --dsr-n-cores-batch 1
```

`--dsr-timeout 0` disables the stage 3 timeout. By default, PyTorch uses
`3600s` and TensorFlow uses `14400s`.

`--dsr-n-cores-batch -1` means use all available CPU cores.

## CUDA

CUDA applies to the PyTorch backend only. The TensorFlow backend runs on CPU in
`.venv_tf`. PyTorch DSO uses CPU by default; pass `--dsr-device cuda` to enable
GPU execution explicitly.

You can verify GPU visibility from the benchmark environment:

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Outputs

- Results are written under `pinn_dsr_bench/results/`.
- PyTorch DSO source lives in `external/deep-symbolic-optimization-pytorch`.
- TensorFlow DSO source lives in `external/deep-symbolic-optimization`.

## Results notes

2026-03-28 test
```
./run_advection_gaussian.sh --num-runs 1 --dsr-backend tensorflow
./run_poisson_polynomial.sh --num-runs 1
```
