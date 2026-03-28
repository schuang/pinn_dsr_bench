from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

from .base import BasePDE


class Poisson2DPolynomialCase(BasePDE):
    case_id = "poisson2d_polynomial"
    _name = "Poisson2D_Polynomial"
    _spatial_domain = [(0.0, 1.0), (0.0, 1.0)]
    _ground_truth_expr = "x1**4 + 1.2*x2**4"

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_inputs(self) -> int:
        return 2

    @property
    def spatial_domain(self) -> List[Tuple[float, float]]:
        return self._spatial_domain

    def ground_truth(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return x1**4 + 1.2 * x2**4

    def source_term(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return -12.0 * x1**2 - 14.4 * x2**2

    def boundary_condition(self, x: np.ndarray) -> np.ndarray:
        return self.ground_truth(x)

    def get_symbol_library(self) -> List[str]:
        return ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "const", "n2", "n4"]

    def get_variable_names(self) -> List[str]:
        return ["x1", "x2"]

    def get_ground_truth_expression(self) -> str:
        return self._ground_truth_expr

    def pde_residual(
        self,
        x: np.ndarray,
        u: np.ndarray,
        du_dx: Callable,
        d2u_dx2: Callable,
    ) -> np.ndarray:
        laplacian = d2u_dx2(x, u, i=0, j=0) + d2u_dx2(x, u, i=1, j=1)
        return -laplacian - self.source_term(x)
