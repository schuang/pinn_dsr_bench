from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

os.environ.setdefault("DEEPXDE_BACKEND", "pytorch")

import deepxde as dde
import numpy as np

from pinn_dsr_bench.pde.advection2d_cases import Advection2DGaussianCase
from pinn_dsr_bench.pde.base import BasePDE
from pinn_dsr_bench.pde.poisson2d_cases import Poisson2DPolynomialCase


@dataclass
class PINNConfig:
    num_hidden_layers: int = 4
    num_neurons_per_layer: int = 50
    activation: str = "tanh"
    initializer: str = "Glorot uniform"
    num_domain: int = 2000
    num_boundary: int = 500
    num_test: int = 1000
    num_initial: int = 500
    adam_epochs: int = 10000
    adam_lr: float = 1e-3
    use_lbfgs: bool = True
    lbfgs_epochs: int = 1000
    seed: Optional[int] = None


class PINNTrainer:
    def __init__(self, pde: BasePDE, config: PINNConfig):
        self.pde = pde
        self.config = config
        self.model = None
        self.data = None
        self._is_trained = False
        if config.seed is not None:
            dde.config.set_random_seed(config.seed)

    def _create_geometry(self):
        spatial = self.pde.spatial_domain
        geom = dde.geometry.Rectangle(
            [spatial[0][0], spatial[1][0]],
            [spatial[0][1], spatial[1][1]],
        )
        if self.pde.is_time_dependent:
            t_lo, t_hi = self.pde.temporal_domain
            geom = dde.geometry.GeometryXTime(geom, dde.geometry.TimeDomain(t_lo, t_hi))
        return geom

    def _create_pde_function(self):
        if isinstance(self.pde, Poisson2DPolynomialCase):
            source_fn = self.pde.source_term

            def pde_residual(x, u):
                d2u_dx1 = dde.grad.hessian(u, x, i=0, j=0)
                d2u_dx2 = dde.grad.hessian(u, x, i=1, j=1)
                laplacian = d2u_dx1 + d2u_dx2
                x_np = x.detach().cpu().numpy() if hasattr(x, "detach") else x
                f_np = source_fn(x_np)
                if hasattr(x, "detach"):
                    import torch

                    f = torch.tensor(f_np, dtype=x.dtype, device=x.device)
                else:
                    f = f_np
                return -laplacian - f

            return pde_residual

        if isinstance(self.pde, Advection2DGaussianCase):
            vx, vy = self.pde.velocity

            def pde_residual(x, u):
                du_dt = dde.grad.jacobian(u, x, i=0, j=2)
                du_dx1 = dde.grad.jacobian(u, x, i=0, j=0)
                du_dx2 = dde.grad.jacobian(u, x, i=0, j=1)
                return du_dt + vx * du_dx1 + vy * du_dx2

            return pde_residual

        raise NotImplementedError(f"Unsupported PDE type: {type(self.pde)}")

    def _create_boundary_conditions(self, geom):
        def bc_func(x):
            return self.pde.boundary_condition(x)

        return [dde.icbc.DirichletBC(geom, bc_func, lambda _, on_boundary: on_boundary)]

    def _create_initial_condition(self, geom):
        if not self.pde.is_time_dependent:
            return None

        def ic_func(x):
            return self.pde.initial_condition(x)

        return dde.icbc.IC(geom, ic_func, lambda _, on_initial: on_initial)

    def _create_network(self):
        layer_sizes = [self.pde.num_inputs]
        layer_sizes.extend([self.config.num_neurons_per_layer] * self.config.num_hidden_layers)
        layer_sizes.append(1)
        return dde.nn.pytorch.FNN(layer_sizes, self.config.activation, self.config.initializer)

    def setup(self) -> None:
        geom = self._create_geometry()
        pde_func = self._create_pde_function()
        bcs = self._create_boundary_conditions(geom)
        if self.pde.is_time_dependent:
            ic = self._create_initial_condition(geom)
            if ic is not None:
                bcs.append(ic)

        if self.pde.is_time_dependent:
            self.data = dde.data.TimePDE(
                geom,
                pde_func,
                bcs,
                num_domain=self.config.num_domain,
                num_boundary=self.config.num_boundary,
                num_initial=self.config.num_initial,
                num_test=self.config.num_test,
            )
        else:
            self.data = dde.data.PDE(
                geom,
                pde_func,
                bcs,
                num_domain=self.config.num_domain,
                num_boundary=self.config.num_boundary,
                num_test=self.config.num_test,
            )

        self.model = dde.Model(self.data, self._create_network())

    def train(self, verbose: bool = True) -> Dict[str, Any]:
        if self.model is None:
            self.setup()

        self.model.compile("adam", lr=self.config.adam_lr)
        _, train_state = self.model.train(
            iterations=self.config.adam_epochs,
            display_every=100 if verbose else self.config.adam_epochs + 1,
        )

        if self.config.use_lbfgs:
            self.model.compile("L-BFGS")
            _, train_state = self.model.train(
                iterations=self.config.lbfgs_epochs,
                display_every=100 if verbose else self.config.lbfgs_epochs + 1,
            )

        self._is_trained = True
        return {
            "final_loss": float(train_state.best_loss_train),
            "best_step": int(train_state.best_step),
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict(x)

    def evaluate(self, n_test: int = 1000, seed: Optional[int] = None) -> Dict[str, float]:
        x_test = self.pde.sample_domain(n_test, seed=seed)
        u_pred = self.predict(x_test)
        u_true = self.pde.ground_truth(x_test)
        mse = np.mean((u_pred - u_true) ** 2)
        l2_error = np.sqrt(np.sum((u_pred - u_true) ** 2)) / np.sqrt(np.sum(u_true ** 2))
        return {
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "l2_error": float(l2_error),
        }

    def generate_dsr_data(
        self,
        n_points: int = 1000,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = self.pde.sample_domain(n_points, seed=seed)
        return x, self.predict(x)
