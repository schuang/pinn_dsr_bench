from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import numpy as np


class BasePDE(ABC):
    @property
    @abstractmethod
    def case_id(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_inputs(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def spatial_domain(self) -> List[Tuple[float, float]]:
        raise NotImplementedError

    @property
    def temporal_domain(self) -> Optional[Tuple[float, float]]:
        return None

    @property
    def is_time_dependent(self) -> bool:
        return self.temporal_domain is not None

    @abstractmethod
    def ground_truth(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def source_term(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def boundary_condition(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def initial_condition(self, x: np.ndarray) -> Optional[np.ndarray]:
        return None

    @abstractmethod
    def get_symbol_library(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_variable_names(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_ground_truth_expression(self) -> str:
        raise NotImplementedError

    def sample_domain(self, n_points: int, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        columns = [rng.uniform(lo, hi, n_points) for lo, hi in self.spatial_domain]
        if self.is_time_dependent:
            t_lo, t_hi = self.temporal_domain
            columns.append(rng.uniform(t_lo, t_hi, n_points))
        return np.column_stack(columns)

    def sample_boundary(self, n_points: int, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n_dim = len(self.spatial_domain)
        points_per_face = n_points // (2 * n_dim) + 1
        faces = []

        for dim, (lo, hi) in enumerate(self.spatial_domain):
            for boundary_val in (lo, hi):
                coords = []
                for current_dim, (dim_lo, dim_hi) in enumerate(self.spatial_domain):
                    if current_dim == dim:
                        coords.append(np.full(points_per_face, boundary_val))
                    else:
                        coords.append(rng.uniform(dim_lo, dim_hi, points_per_face))
                if self.is_time_dependent:
                    t_lo, t_hi = self.temporal_domain
                    coords.append(rng.uniform(t_lo, t_hi, points_per_face))
                faces.append(np.column_stack(coords))

        boundary_points = np.vstack(faces)
        indices = rng.choice(boundary_points.shape[0], size=n_points, replace=False)
        return boundary_points[indices]

    def sample_initial(self, n_points: int, seed: Optional[int] = None) -> Optional[np.ndarray]:
        if not self.is_time_dependent:
            return None
        rng = np.random.default_rng(seed)
        columns = [rng.uniform(lo, hi, n_points) for lo, hi in self.spatial_domain]
        t_lo, _ = self.temporal_domain
        columns.append(np.full(n_points, t_lo))
        return np.column_stack(columns)

    def pde_residual(
        self,
        x: np.ndarray,
        u: np.ndarray,
        du_dx: Callable,
        d2u_dx2: Optional[Callable],
    ) -> np.ndarray:
        raise NotImplementedError
