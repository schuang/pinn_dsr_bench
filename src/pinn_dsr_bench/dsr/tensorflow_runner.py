from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def create_dso_config(
    n_epochs: int,
    learning_rate: float,
    entropy_weight: float,
    risk_factor: float,
    batch_size: int,
    operators,
    seed,
    n_cores_batch,
):
    return {
        "experiment": {
            "seed": seed,
            "logdir": None,
            "exp_name": None,
        },
        "task": {
            "task_type": "regression",
            "function_set": operators,
            "metric": "inv_nrmse",
            "metric_params": [1.0],
            "threshold": 1e-12,
            "protected": False,
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
        "prior": {
            "length": {"min_": 4, "max_": 30, "on": True},
            "repeat": {"tokens": "const", "min_": None, "max_": 3, "on": True},
            "inverse": {"on": True},
            "trig": {"on": True},
            "const": {"on": True},
            "no_inputs": {"on": True},
            "soft_length": {"loc": 10, "scale": 5, "on": True},
        },
        "controller": {
            "cell": "lstm",
            "num_layers": 1,
            "num_units": 32,
            "initializer": "zeros",
            "embedding": False,
            "embedding_size": 8,
            "optimizer": "adam",
            "learning_rate": learning_rate,
            "entropy_weight": entropy_weight,
            "entropy_gamma": 0.7,
            "ppo": False,
            "ppo_clip_ratio": 0.2,
            "ppo_n_iters": 10,
            "ppo_n_mb": 4,
            "pqt": False,
            "pqt_k": 10,
            "pqt_batch_size": 1,
            "pqt_weight": 200.0,
            "pqt_use_pg": False,
        },
        "gp_meld": {"run_gp_meld": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="TensorFlow DSO runner")
    parser.add_argument("input_csv")
    parser.add_argument("output_json")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--entropy-weight", type=float, default=0.07)
    parser.add_argument("--risk-factor", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--operators", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-cores-batch", type=int, default=-1)
    args = parser.parse_args()

    try:
        import numpy as np
        import pandas as pd
        import tensorflow as tf
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
                    "error": f"Failed to import TensorFlow DSO stack: {exc}",
                    "profile": {
                        "backend": "tensorflow",
                        "device": "cpu",
                        "n_cores_batch": 1,
                    },
                },
                handle,
                indent=2,
            )
        return 1

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    if args.seed is not None:
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

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
        nmse = float(((((y - y_pred) ** 2).mean()) / y.var())) if y.var() > 0 else float("inf")
        rmse = float((((y - y_pred) ** 2).mean()) ** 0.5)
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
                "backend": "tensorflow",
                "device": "cpu",
                "n_cores_batch": config["training"]["n_cores_batch"],
            },
        }
        print(
            "DSO profile: "
            f"backend={output['profile']['backend']}, "
            f"device={output['profile']['device']}, "
            f"n_cores_batch={output['profile']['n_cores_batch']}, "
            f"fit={fit_time:.2f}s, "
            f"predict={predict_time:.2f}s"
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
                "backend": "tensorflow",
                "device": "cpu",
                "n_cores_batch": config["training"]["n_cores_batch"],
            },
        }
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)

    with open(args.output_json, "w") as handle:
        json.dump(output, handle, indent=2)
    return 0 if output["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
