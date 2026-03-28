from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from pinn_dsr_bench.benchmark.metrics import evaluate_expression
from pinn_dsr_bench.dsr import DSRConfig, DSRWrapper
from pinn_dsr_bench.pde.base import BasePDE
from pinn_dsr_bench.pinn import PINNConfig, PINNTrainer


@dataclass
class BenchmarkConfig:
    pinn_num_hidden_layers: int = 4
    pinn_num_neurons: int = 50
    pinn_activation: str = "tanh"
    pinn_num_domain: int = 2000
    pinn_num_boundary: int = 500
    pinn_adam_epochs: int = 10000
    pinn_adam_lr: float = 1e-3
    pinn_use_lbfgs: bool = True
    pinn_lbfgs_epochs: int = 1000
    dsr_max_epochs: int = 200
    dsr_batch_size: int = 1000
    dsr_learning_rate: float = 0.001
    dsr_entropy_weight: float = 0.07
    dsr_risk_factor: float = 0.05
    dsr_num_samples: int = 1000
    dsr_python: Optional[str] = None
    dsr_backend: str = "pytorch"
    dsr_device: Optional[str] = None
    dsr_timeout_sec: Optional[int] = None
    dsr_n_cores_batch: Optional[int] = None
    num_runs: int = 1
    base_seed: int = 42
    output_dir: Optional[str] = None


class BenchmarkRunner:
    def __init__(self, pde: BasePDE, config: BenchmarkConfig):
        self.pde = pde
        self.config = config
        self.results: List[Dict[str, Any]] = []
        if self.config.output_dir is None:
            self.config.output_dir = os.path.join("results", pde.case_id)

    def _create_pinn_config(self, seed: int) -> PINNConfig:
        return PINNConfig(
            num_hidden_layers=self.config.pinn_num_hidden_layers,
            num_neurons_per_layer=self.config.pinn_num_neurons,
            activation=self.config.pinn_activation,
            num_domain=self.config.pinn_num_domain,
            num_boundary=self.config.pinn_num_boundary,
            adam_epochs=self.config.pinn_adam_epochs,
            adam_lr=self.config.pinn_adam_lr,
            use_lbfgs=self.config.pinn_use_lbfgs,
            lbfgs_epochs=self.config.pinn_lbfgs_epochs,
            seed=seed,
        )

    def _create_dsr_config(self, seed: int) -> DSRConfig:
        return DSRConfig(
            operators=self.pde.get_symbol_library(),
            learning_rate=self.config.dsr_learning_rate,
            entropy_weight=self.config.dsr_entropy_weight,
            risk_factor=self.config.dsr_risk_factor,
            max_epochs=self.config.dsr_max_epochs,
            batch_size=self.config.dsr_batch_size,
            seed=seed,
            python_bin=self.config.dsr_python,
            backend=self.config.dsr_backend,
            device=self.config.dsr_device,
            timeout_sec=self.config.dsr_timeout_sec,
            n_cores_batch=self.config.dsr_n_cores_batch,
        )

    def run_single(self, run_id: int, seed: int, verbose: bool = True) -> Dict[str, Any]:
        result = {
            "run_id": run_id,
            "seed": seed,
            "pde_name": self.pde.name,
            "timestamp": datetime.now().isoformat(),
            "ground_truth_expression": self.pde.get_ground_truth_expression(),
        }

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Run {run_id + 1}/{self.config.num_runs} (seed={seed})")
            print(f"{'=' * 60}")

        trainer = PINNTrainer(self.pde, self._create_pinn_config(seed))

        if verbose:
            print("\n[Stage 1] Training native PINN...")
        start = time.time()
        trainer.train(verbose=verbose)
        result["pinn_training_time"] = time.time() - start
        pinn_metrics = trainer.evaluate(seed=seed)
        result["pinn_l2_error"] = pinn_metrics["l2_error"]
        result["pinn_mse"] = pinn_metrics["mse"]

        if verbose:
            print(f"PINN L2 error: {result['pinn_l2_error']:.6e}")

        if verbose:
            print(f"\n[Stage 2] Generating DSR data ({self.config.dsr_num_samples} points)...")
        x, u = trainer.generate_dsr_data(n_points=self.config.dsr_num_samples, seed=seed)

        if verbose:
            print("\n[Stage 3] Running native DSO...")
        dsr = DSRWrapper(self._create_dsr_config(seed), self.pde.get_variable_names())
        start = time.time()
        expr_str, reward = dsr.fit(x, u, verbose=verbose)
        result["dsr_time"] = time.time() - start
        result["expression"] = expr_str
        result["dsr_reward"] = reward
        if dsr.last_payload is not None and "profile" in dsr.last_payload:
            result["dsr_profile"] = dsr.last_payload["profile"]

        if verbose:
            print("\n[Stage 4] Evaluating expression...")
        evaluation = evaluate_expression(expr_str, self.pde, seed=seed)
        result.update(evaluation)
        return result

    def run_all(self, verbose: bool = True) -> List[Dict[str, Any]]:
        self.results = []
        for run_id in range(self.config.num_runs):
            seed = self.config.base_seed + run_id
            self.results.append(self.run_single(run_id, seed, verbose=verbose))
        return self.results

    def compute_statistics(self) -> Dict[str, Any]:
        if not self.results:
            return {}

        l_phy_values = [r["l_phy"] for r in self.results if np.isfinite(r.get("l_phy", np.inf))]
        pre_values = [bool(r.get("pre", False)) for r in self.results]
        pinn_l2_values = [r["pinn_l2_error"] for r in self.results if np.isfinite(r.get("pinn_l2_error", np.inf))]
        mse_values = [r["mse"] for r in self.results if np.isfinite(r.get("mse", np.inf))]

        stats: Dict[str, Any] = {
            "num_runs": len(self.results),
            "num_successful": len(l_phy_values),
            "pre_rate": float(np.mean(pre_values) * 100.0) if pre_values else 0.0,
            "pre_count": int(sum(pre_values)),
        }

        if l_phy_values:
            arr = np.asarray(l_phy_values)
            stats["l_phy_mean"] = float(np.mean(arr))
            stats["l_phy_std"] = float(np.std(arr))
            stats["l_phy_min"] = float(np.min(arr))
            stats["l_phy_max"] = float(np.max(arr))

        if pinn_l2_values:
            arr = np.asarray(pinn_l2_values)
            stats["pinn_l2_mean"] = float(np.mean(arr))
            stats["pinn_l2_std"] = float(np.std(arr))

        if mse_values:
            arr = np.asarray(mse_values)
            stats["mse_mean"] = float(np.mean(arr))
            stats["mse_min"] = float(np.min(arr))
            stats["mse_max"] = float(np.max(arr))

        return stats

    def save_results(self) -> str:
        os.makedirs(self.config.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.config.output_dir, f"results_{self.pde.name}_{timestamp}.json")
        payload = {
            "pde_name": self.pde.name,
            "case_id": self.pde.case_id,
            "ground_truth_expression": self.pde.get_ground_truth_expression(),
            "config": asdict(self.config),
            "statistics": self.compute_statistics(),
            "runs": self.results,
        }
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2, default=str)
        print(f"\nResults saved to: {path}")
        return path

    def print_summary(self) -> None:
        stats = self.compute_statistics()
        print("\n" + "=" * 60)
        print(f"SUMMARY: {self.pde.name}")
        print("=" * 60)
        print(f"Runs: {stats.get('num_successful', 0)}/{stats.get('num_runs', 0)} successful")
        if "l_phy_mean" in stats:
            print(f"L_PHY mean: {stats['l_phy_mean']:.4e}")
        if "pre_rate" in stats:
            print(f"PRE rate: {stats['pre_rate']:.1f}%")
        if "pinn_l2_mean" in stats:
            print(f"PINN L2 mean: {stats['pinn_l2_mean']:.4e}")
