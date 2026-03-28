#!/usr/bin/env python3

import argparse
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT_DIR, ".cache", "matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from pinn_dsr_bench.pde import get_case, list_cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run native PINN + DSR benchmark cases.")
    parser.add_argument("--case", choices=list_cases(), help="Case identifier to run.")
    parser.add_argument("--list-cases", action="store_true", help="List supported cases and exit.")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of independent runs.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--output-dir", type=str, default=None, help="Result directory.")
    parser.add_argument("--pinn-domain-points", type=int, default=2000, help="PINN domain points.")
    parser.add_argument("--pinn-boundary-points", type=int, default=500, help="PINN boundary points.")
    parser.add_argument("--dsr-samples", type=int, default=1000, help="PINN samples passed to DSR.")
    parser.add_argument("--pinn-adam-epochs", type=int, default=10000, help="PINN Adam epochs.")
    parser.add_argument("--pinn-lbfgs-epochs", type=int, default=1000, help="PINN L-BFGS epochs.")
    parser.add_argument("--no-lbfgs", action="store_true", help="Disable PINN L-BFGS refinement.")
    parser.add_argument("--dsr-epochs", type=int, default=200, help="DSO epochs.")
    parser.add_argument(
        "--dsr-python",
        type=str,
        default=os.environ.get("PINN_DSR_BENCH_DSO_PYTHON"),
        help="Python executable for native DSO environment.",
    )
    parser.add_argument(
        "--dsr-backend",
        type=str,
        choices=("pytorch", "tensorflow"),
        default=os.environ.get("PINN_DSR_BENCH_DSO_BACKEND", "pytorch"),
        help="Native DSO backend to use.",
    )
    parser.add_argument(
        "--dsr-device",
        type=str,
        choices=("cpu", "cuda"),
        default=os.environ.get("PINN_DSR_BENCH_DSO_DEVICE"),
        help="Execution device for native DSO.",
    )
    parser.add_argument(
        "--dsr-timeout",
        type=int,
        default=None,
        help="Stage 3 timeout in seconds. Use 0 to disable the timeout.",
    )
    parser.add_argument(
        "--dsr-n-cores-batch",
        type=int,
        default=None,
        help="DSO batch evaluation worker count. Use -1 for all CPU cores.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce logging.")
    parser.add_argument("--quick-test", action="store_true", help="Run a light smoke test configuration.")
    return parser


def resolve_display_dsr_python(root_dir: str, backend: str, dsr_python: str | None) -> str:
    if dsr_python:
        return dsr_python
    if backend == "tensorflow":
        tf_python = os.path.join(root_dir, ".venv_tf", "bin", "python")
        if os.path.exists(tf_python):
            return tf_python
    return sys.executable


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_cases:
        for case_id in list_cases():
            case = get_case(case_id)
            print(f"{case_id}: {case.get_ground_truth_expression()}")
        return 0

    if not args.case:
        parser.error("--case is required unless --list-cases is used")

    from pinn_dsr_bench.benchmark.runner import BenchmarkConfig, BenchmarkRunner

    config = BenchmarkConfig(
        num_runs=args.num_runs,
        base_seed=args.seed,
        output_dir=args.output_dir,
        pinn_num_domain=args.pinn_domain_points,
        pinn_num_boundary=args.pinn_boundary_points,
        pinn_adam_epochs=args.pinn_adam_epochs,
        pinn_lbfgs_epochs=args.pinn_lbfgs_epochs,
        pinn_use_lbfgs=not args.no_lbfgs,
        dsr_num_samples=args.dsr_samples,
        dsr_max_epochs=args.dsr_epochs,
        dsr_python=args.dsr_python,
        dsr_backend=args.dsr_backend,
        dsr_device=args.dsr_device,
        dsr_timeout_sec=args.dsr_timeout,
        dsr_n_cores_batch=args.dsr_n_cores_batch,
    )

    if args.quick_test:
        config.num_runs = 1
        config.pinn_adam_epochs = 1000
        config.pinn_use_lbfgs = False
        config.dsr_max_epochs = 10

    case = get_case(args.case)
    if config.output_dir is None:
        config.output_dir = os.path.join(ROOT_DIR, "results", args.case)

    print("Configuration:")
    print(f"  Case: {args.case}")
    print(f"  PDE: {case.name}")
    print(f"  Ground truth: {case.get_ground_truth_expression()}")
    print(f"  Runs: {config.num_runs}")
    print(f"  PINN domain points: {config.pinn_num_domain}")
    print(f"  PINN boundary points: {config.pinn_num_boundary}")
    print(f"  PINN Adam epochs: {config.pinn_adam_epochs}")
    print(f"  PINN L-BFGS: {config.pinn_use_lbfgs}")
    print(f"  DSR epochs: {config.dsr_max_epochs}")
    print(f"  DSR samples: {config.dsr_num_samples}")
    print(f"  DSR backend: {config.dsr_backend}")
    print(f"  DSR python: {resolve_display_dsr_python(ROOT_DIR, config.dsr_backend, config.dsr_python)}")
    print(f"  DSR device: {config.dsr_device or 'cpu'}")
    print(f"  DSR timeout: {config.dsr_timeout_sec if config.dsr_timeout_sec is not None else ('14400' if config.dsr_backend == 'tensorflow' else '3600')}s")
    print(f"  DSR n_cores_batch: {config.dsr_n_cores_batch if config.dsr_n_cores_batch is not None else 'backend default'}")
    print(f"  Output dir: {config.output_dir}")

    runner = BenchmarkRunner(case, config)
    runner.run_all(verbose=not args.quiet)
    runner.print_summary()
    runner.save_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
