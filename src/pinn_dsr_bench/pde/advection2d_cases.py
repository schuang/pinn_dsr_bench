from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

from .base import BasePDE


class Advection2DGaussianCase(BasePDE):
    case_id = "advection2d_gaussian"
    _name = "Advection2D_Gaussian"
    _spatial_domain = [(0.0, 1.0), (0.0, 1.0)]
    _temporal_domain = (0.0, 1.0)
    _ground_truth_expr = "exp(-((x1 - t)**2 + (x2 - t)**2) / 0.5)"
    _velocity = (1.0, 1.0)

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_inputs(self) -> int:
        return 3

    @property
    def spatial_domain(self) -> List[Tuple[float, float]]:
        return self._spatial_domain

    @property
    def temporal_domain(self) -> Tuple[float, float]:
        return self._temporal_domain

    @property
    def velocity(self) -> Tuple[float, float]:
        return self._velocity

    def ground_truth(self, x: np.ndarray) -> np.ndarray:
        x1, x2, t = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return np.exp(-((x1 - t) ** 2 + (x2 - t) ** 2) / 0.5)

    def source_term(self, x: np.ndarray) -> np.ndarray:
        return np.zeros((x.shape[0], 1))

    def boundary_condition(self, x: np.ndarray) -> np.ndarray:
        return self.ground_truth(x)

    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return np.exp(-(x1**2 + x2**2) / 0.5)

    def get_symbol_library(self) -> List[str]:
        return ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "const"] 

    def get_variable_names(self) -> List[str]:
        return ["x1", "x2", "t"]

    def get_ground_truth_expression(self) -> str:
        return self._ground_truth_expr

    def pde_residual(
        self,
        x: np.ndarray,
        u: np.ndarray,
        du_dx: Callable,
        d2u_dx2: Callable,
    ) -> np.ndarray:
        del d2u_dx2
        vx, vy = self.velocity
        du_dt = du_dx(x, u, i=2)
        du_dx1 = du_dx(x, u, i=0)
        du_dx2 = du_dx(x, u, i=1)
        return du_dt + vx * du_dx1 + vy * du_dx2
