from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import warnings

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PYTORCH_DSO_SRC = os.path.join(
    ROOT_DIR,
    "external",
    "deep-symbolic-optimization-pytorch",
    "dso",
)
if PYTORCH_DSO_SRC not in sys.path:
    sys.path.insert(0, PYTORCH_DSO_SRC)


def create_dso_config(
    n_epochs: int,
    learning_rate: float,
    entropy_weight: float,
    risk_factor: float,
    batch_size: int,
    operators,
    seed,
    device,
    n_cores_batch,
):
    if device is None:
        device = "cpu"
    return {
        "experiment": {
            "seed": seed,
            "logdir": None,
            "exp_name": None,
            "device": device,
        },
        "task": {
            "task_type": "regression",
            "function_set": operators,
            "metric": "inv_nrmse",
            "metric_params": [1.0],
            "threshold": 1e-12,
            "protected": True,
            "poly_optimizer_params": {
                "degree": 3,
                "coef_tol": 1e-6,
                "regressor": "dso_least_squares",
            },
        },
        "training": {
            "n_samples": n_epochs * batch_size,
            "batch_size": batch_size,
            "epsilon": risk_factor,
            "baseline": "R_e",
            "b_jumpstart": False,
            "n_cores_batch": n_cores_batch,
            "complexity": "length",
            "const_optimizer": "scipy",
            "const_params": {"method": "L-BFGS-B", "options": {"maxiter": 1000}},
            "alpha": 0.5,
            "verbose": True,
            "early_stopping": False,
        },
        "logging": {
            "save_all_iterations": False,
            "save_summary": False,
            "save_positional_entropy": False,
            "save_pareto_front": False,
            "save_cache": False,
            "save_freq": 1,
            "hof": 20,
        },
        "state_manager": {
            "type": "hierarchical",
            "observe_action": False,
            "observe_parent": True,
            "observe_sibling": True,
            "observe_dangling": False,
            "embedding": False,
            "embedding_size": 8,
        },
        "policy": {
            "policy_type": "rnn",
            "max_length": 30,
            "cell": "lstm",
            "num_layers": 1,
            "num_units": 32,
            "initializer": "zeros",
        },
        "policy_optimizer": {
            "policy_optimizer_type": "pg",
            "summary": False,
            "optimizer": "adam",
            "learning_rate": learning_rate,
            "entropy_weight": entropy_weight,
            "entropy_gamma": 0.7,
        },
        "prior": {
            "length": {"min_": 4, "max_": 30, "on": True},
            "repeat": {"tokens": "const", "min_": None, "max_": 3, "on": True},
            "inverse": {"on": True},
            "trig": {"on": True},
            "const": {"on": True},
            "no_inputs": {"on": True},
            "uniform_arity": {"on": False},
            "soft_length": {"loc": 10, "scale": 5, "on": True},
        },
        "gp_meld": {"run_gp_meld": False},
        "checkpoint": {},
    }


def _install_dso_profile_hooks():
    import atexit
    from dso.program import Program

    profile = {
        "execute_calls": 0,
        "execute_time_sec": 0.0,
        "optimize_calls": 0,
        "optimize_time_sec": 0.0,
    }

    original_execute = Program.execute
    original_optimize = Program.optimize

    def profiled_execute(self, X):
        start = time.perf_counter()
        try:
            return original_execute(self, X)
        finally:
            profile["execute_calls"] += 1
            profile["execute_time_sec"] += time.perf_counter() - start

    def profiled_optimize(self):
        start = time.perf_counter()
        try:
            return original_optimize(self)
        finally:
            profile["optimize_calls"] += 1
            profile["optimize_time_sec"] += time.perf_counter() - start

    Program.execute = profiled_execute
    Program.optimize = profiled_optimize

    def restore_hooks():
        Program.execute = original_execute
        Program.optimize = original_optimize

    atexit.register(restore_hooks)
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description="Native DSO runner")
    parser.add_argument("input_csv")
    parser.add_argument("output_json")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--entropy-weight", type=float, default=0.07)
    parser.add_argument("--risk-factor", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--operators", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, choices=("cpu", "cuda"), default=None)
    parser.add_argument("--n-cores-batch", type=int, default=-1)
    args = parser.parse_args()

    try:
        import numpy as np
        import pandas as pd
        from dso import DeepSymbolicRegressor
    except Exception as exc:
        with open(args.output_json, "w") as handle:
            json.dump(
                {
                    "expression": None,
                    "reward": 0.0,
                    "nmse": float("inf"),
                    "rmse": float("inf"),
                    "complexity": None,
                    "success": False,
                    "error": f"Failed to import PyTorch DSO stack: {exc}",
                },
                handle,
                indent=2,
            )
        return 1

    if args.seed is not None:
        np.random.seed(args.seed)

    profile = _install_dso_profile_hooks()

    frame = pd.read_csv(args.input_csv)
    x = frame.iloc[:, :-1].values
    y = frame.iloc[:, -1].values

    config = create_dso_config(
        n_epochs=args.epochs,
        learning_rate=args.learning_rate,
        entropy_weight=args.entropy_weight,
        risk_factor=args.risk_factor,
        batch_size=args.batch_size,
        operators=args.operators.split(","),
        seed=args.seed,
        device=args.device,
        n_cores_batch=args.n_cores_batch,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
        json.dump(config, handle)
        config_path = handle.name

    try:
        model = DeepSymbolicRegressor(config_path)
        fit_start = time.perf_counter()
        model.fit(x, y)
        fit_time = time.perf_counter() - fit_start
        predict_start = time.perf_counter()
        y_pred = model.predict(x)
        predict_time = time.perf_counter() - predict_start
        nmse = float(np.mean((y - y_pred) ** 2) / np.var(y)) if np.var(y) > 0 else float("inf")
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        output = {
            "expression": str(model.program_.sympy_expr),
            "reward": float(model.program_.r),
            "nmse": nmse,
            "rmse": rmse,
            "complexity": int(model.program_.complexity),
            "success": True,
            "error": None,
            "profile": {
                "fit_time_sec": fit_time,
                "predict_time_sec": predict_time,
                "execute_calls": profile["execute_calls"],
                "execute_time_sec": profile["execute_time_sec"],
                "optimize_calls": profile["optimize_calls"],
                "optimize_time_sec": profile["optimize_time_sec"],
                "backend": "pytorch",
                "device": config["experiment"]["device"],
                "n_cores_batch": config["training"]["n_cores_batch"],
            },
        }
        print(
            "DSO profile: "
            f"device={output['profile']['device']}, "
            f"n_cores_batch={output['profile']['n_cores_batch']}, "
            f"fit={fit_time:.2f}s, "
            f"predict={predict_time:.2f}s, "
            f"execute_calls={profile['execute_calls']}, "
            f"execute_time={profile['execute_time_sec']:.2f}s, "
            f"optimize_calls={profile['optimize_calls']}, "
            f"optimize_time={profile['optimize_time_sec']:.2f}s"
        )
    except Exception as exc:
        output = {
            "expression": None,
            "reward": 0.0,
            "nmse": float("inf"),
            "rmse": float("inf"),
            "complexity": None,
            "success": False,
            "error": str(exc),
            "profile": {
                "execute_calls": profile["execute_calls"],
                "execute_time_sec": profile["execute_time_sec"],
                "optimize_calls": profile["optimize_calls"],
                "optimize_time_sec": profile["optimize_time_sec"],
                "backend": "pytorch",
                "device": config["experiment"]["device"],
                "n_cores_batch": config["training"]["n_cores_batch"],
            },
        }

    with open(args.output_json, "w") as handle:
        json.dump(output, handle, indent=2)
    return 0 if output["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
