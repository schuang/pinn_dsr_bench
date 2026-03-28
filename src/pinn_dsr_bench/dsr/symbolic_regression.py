from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@dataclass
class DSRConfig:
    operators: List[str] = field(default_factory=lambda: ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"])
    learning_rate: float = 0.001
    entropy_weight: float = 0.07
    risk_factor: float = 0.05
    max_epochs: int = 200
    batch_size: int = 1000
    seed: Optional[int] = None
    python_bin: Optional[str] = None
    backend: str = "pytorch"
    device: Optional[str] = None
    timeout_sec: Optional[int] = None
    n_cores_batch: Optional[int] = None


class DSRWrapper:
    def __init__(self, config: DSRConfig, variable_names: List[str]):
        self.config = config
        self.variable_names = variable_names
        self.last_payload: Optional[dict] = None

    def _create_dataset_file(self, x: np.ndarray, y: np.ndarray, filepath: str) -> None:
        y_flat = y.flatten() if y.ndim > 1 else y
        data = np.column_stack([x, y_flat])
        header = ",".join(self.variable_names + ["y"])
        np.savetxt(filepath, data, delimiter=",", header=header, comments="")

    def _normalize_expression_variables(self, expression: Optional[str]) -> Optional[str]:
        if expression is None:
            return None

        normalized = expression
        for index, variable_name in enumerate(self.variable_names, start=1):
            generic_name = f"x{index}"
            if generic_name == variable_name:
                continue
            normalized = re.sub(rf"\b{re.escape(generic_name)}\b", variable_name, normalized)
        return normalized

    def _resolve_runner(self) -> str:
        if self.config.backend == "tensorflow":
            return os.path.join(os.path.dirname(__file__), "tensorflow_runner.py")
        return os.path.join(os.path.dirname(__file__), "native_runner.py")

    def _resolve_python(self) -> str:
        if self.config.python_bin:
            return self.config.python_bin

        env_python = os.environ.get("PINN_DSR_BENCH_DSO_PYTHON")
        if env_python:
            return env_python

        if self.config.backend == "tensorflow":
            tf_python = os.path.join(ROOT_DIR, ".venv_tf", "bin", "python")
            if os.path.exists(tf_python):
                return tf_python

        return sys.executable

    def _resolve_timeout(self) -> Optional[int]:
        env_timeout = os.environ.get("PINN_DSR_BENCH_DSO_TIMEOUT")
        if env_timeout is not None and env_timeout != "":
            parsed = int(env_timeout)
            return None if parsed <= 0 else parsed

        if self.config.timeout_sec is not None:
            return None if self.config.timeout_sec <= 0 else self.config.timeout_sec

        if self.config.backend == "tensorflow":
            return 4 * 3600
        return 3600

    def _resolve_n_cores_batch(self) -> Optional[int]:
        env_value = os.environ.get("PINN_DSR_BENCH_DSO_N_CORES_BATCH")
        if env_value is not None and env_value != "":
            return int(env_value)
        return self.config.n_cores_batch

    def fit(self, x: np.ndarray, y: np.ndarray, verbose: bool = True) -> Tuple[Optional[str], float]:
        runner_py = self._resolve_runner()
        project_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        python_bin = self._resolve_python()
        timeout_sec = self._resolve_timeout()
        n_cores_batch = self._resolve_n_cores_batch()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_csv = os.path.join(tmpdir, "input.csv")
            output_json = os.path.join(tmpdir, "output.json")
            self._create_dataset_file(x, y, input_csv)

            env = os.environ.copy()
            env["PYTHONPATH"] = project_src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

            cmd = [
                python_bin,
                runner_py,
                input_csv,
                output_json,
                "--epochs",
                str(self.config.max_epochs),
                "--learning-rate",
                str(self.config.learning_rate),
                "--entropy-weight",
                str(self.config.entropy_weight),
                "--risk-factor",
                str(self.config.risk_factor),
                "--batch-size",
                str(self.config.batch_size),
                "--operators",
                ",".join(self.config.operators),
            ]
            if self.config.seed is not None:
                cmd.extend(["--seed", str(self.config.seed)])
            if self.config.device and self.config.backend == "pytorch":
                cmd.extend(["--device", self.config.device])
            if n_cores_batch is not None:
                cmd.extend(["--n-cores-batch", str(n_cores_batch)])

            if verbose:
                print(f"Running native DSO backend '{self.config.backend}' with interpreter: {python_bin}")
                timeout_desc = "none" if timeout_sec is None else f"{timeout_sec}s"
                print(f"DSO timeout: {timeout_desc}")
                print(f"DSO n_cores_batch: {n_cores_batch if n_cores_batch is not None else 'runner default'}")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired as exc:
                stdout_tail = (exc.stdout or "").strip()[-1500:]
                stderr_tail = (exc.stderr or "").strip()[-1500:]
                details = [
                    f"DSO backend '{self.config.backend}' timed out after {timeout_sec}s."
                ]
                if stdout_tail:
                    details.append(f"Last stdout:\n{stdout_tail}")
                if stderr_tail:
                    details.append(f"Last stderr:\n{stderr_tail}")
                details.append(
                    "Increase --dsr-timeout, set PINN_DSR_BENCH_DSO_TIMEOUT, or reduce --dsr-epochs/--dsr-samples."
                )
                raise RuntimeError("\n\n".join(details)) from exc
            if verbose and result.stdout.strip():
                print(result.stdout.strip())
            if result.returncode != 0 and verbose and result.stderr.strip():
                print(result.stderr.strip())

            if not os.path.exists(output_json):
                error_detail = result.stderr.strip() or result.stdout.strip() or "unknown subprocess failure"
                raise RuntimeError(f"DSO runner did not produce output JSON: {error_detail}")

            with open(output_json, "r") as handle:
                payload = json.load(handle)
            self.last_payload = payload

            if not payload.get("success"):
                raise RuntimeError(payload.get("error") or "Native DSO failed")

            expression = self._normalize_expression_variables(payload["expression"])
            return expression, float(payload["reward"])
