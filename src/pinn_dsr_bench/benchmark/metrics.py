from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import sympy as sp

from pinn_dsr_bench.pde.advection2d_cases import Advection2DGaussianCase
from pinn_dsr_bench.pde.base import BasePDE
from pinn_dsr_bench.pde.poisson2d_cases import Poisson2DPolynomialCase


def compute_pre(
    expr: sp.Expr,
    pde: BasePDE,
    n_test: int = 1000,
    mse_threshold: float = 1e-8,
    seed: Optional[int] = None,
) -> bool:
    symbols = {name: sp.Symbol(name) for name in pde.get_variable_names()}
    gt_expr = sp.sympify(pde.get_ground_truth_expression(), locals=symbols)
    x_test = pde.sample_domain(n_test, seed=seed)

    try:
        expr_func = sp.lambdify(list(symbols.values()), expr, modules=["numpy"])
        gt_func = sp.lambdify(list(symbols.values()), gt_expr, modules=["numpy"])
        args = [x_test[:, i] for i in range(len(symbols))]
        u_expr = expr_func(*args)
        u_gt = gt_func(*args)
        if isinstance(u_expr, (int, float)):
            u_expr = np.full(len(x_test), u_expr)
        if isinstance(u_gt, (int, float)):
            u_gt = np.full(len(x_test), u_gt)
        u_expr = np.asarray(u_expr, dtype=float)
        u_gt = np.asarray(u_gt, dtype=float)
        if not np.all(np.isfinite(u_expr)):
            return False
        mse = np.mean((u_expr - u_gt) ** 2)
    except Exception:
        return False

    return bool(np.isfinite(mse) and mse < mse_threshold)


def compute_l_phy(
    expr: sp.Expr,
    pde: BasePDE,
    n_domain: int = 1000,
    n_boundary: int = 500,
    seed: Optional[int] = None,
) -> float:
    symbols = {name: sp.Symbol(name) for name in pde.get_variable_names()}
    try:
        expr_func = sp.lambdify(list(symbols.values()), expr, modules=["numpy"])
    except Exception:
        return float("inf")
    x_domain = pde.sample_domain(n_domain, seed=seed)
    x_boundary = pde.sample_boundary(n_boundary, seed=seed)

    if isinstance(pde, Poisson2DPolynomialCase):
        x1, x2 = symbols["x1"], symbols["x2"]
        residual_expr = -(sp.diff(expr, x1, 2) + sp.diff(expr, x2, 2))
        try:
            residual_func = sp.lambdify([x1, x2], residual_expr, modules=["numpy"])
            residual_vals = residual_func(x_domain[:, 0], x_domain[:, 1]) - pde.source_term(x_domain).flatten()
        except Exception:
            return float("inf")
        norm_f = np.mean(pde.source_term(x_domain).flatten() ** 2)
    elif isinstance(pde, Advection2DGaussianCase):
        x1, x2, t = symbols["x1"], symbols["x2"], symbols["t"]
        vx, vy = pde.velocity
        residual_expr = sp.diff(expr, t) + vx * sp.diff(expr, x1) + vy * sp.diff(expr, x2)
        try:
            residual_func = sp.lambdify([x1, x2, t], residual_expr, modules=["numpy"])
            residual_vals = residual_func(x_domain[:, 0], x_domain[:, 1], x_domain[:, 2])
            du_dt_func = sp.lambdify([x1, x2, t], sp.diff(expr, t), modules=["numpy"])
            du_dt_vals = du_dt_func(x_domain[:, 0], x_domain[:, 1], x_domain[:, 2])
        except Exception:
            return float("inf")
        norm_f = np.mean(np.asarray(du_dt_vals) ** 2)
    else:
        raise NotImplementedError(f"Unsupported PDE type: {type(pde)}")

    if isinstance(residual_vals, (int, float)):
        residual_vals = np.full(n_domain, residual_vals)
    norm_f = max(float(norm_f), 1.0)
    residual_vals = np.asarray(residual_vals, dtype=float)
    residual_vals = np.nan_to_num(residual_vals, nan=np.inf, posinf=np.inf, neginf=np.inf)
    mse_f = float(np.mean(residual_vals ** 2))

    boundary_args = [x_boundary[:, i] for i in range(x_boundary.shape[1])]
    try:
        u_boundary = expr_func(*boundary_args)
    except Exception:
        return float("inf")
    if isinstance(u_boundary, (int, float)):
        u_boundary = np.full(n_boundary, u_boundary)
    u_true_boundary = pde.boundary_condition(x_boundary).flatten()
    u_boundary = np.asarray(u_boundary, dtype=float)
    u_boundary = np.nan_to_num(u_boundary, nan=np.inf, posinf=np.inf, neginf=np.inf)
    mse_b = float(np.mean((u_boundary - u_true_boundary) ** 2))
    norm_b = max(float(np.mean(u_true_boundary**2)), 1.0)

    rel_l2_f = np.sqrt(mse_f) / np.sqrt(norm_f)
    rel_l2_b = np.sqrt(mse_b) / np.sqrt(norm_b)

    if pde.is_time_dependent:
        x_initial = pde.sample_initial(n_boundary, seed=seed)
        init_args = [x_initial[:, i] for i in range(x_initial.shape[1])]
        try:
            u_initial = expr_func(*init_args)
        except Exception:
            return float("inf")
        if isinstance(u_initial, (int, float)):
            u_initial = np.full(n_boundary, u_initial)
        u_true_initial = pde.initial_condition(x_initial).flatten()
        u_initial = np.asarray(u_initial, dtype=float)
        u_initial = np.nan_to_num(u_initial, nan=np.inf, posinf=np.inf, neginf=np.inf)
        mse_i = float(np.mean((u_initial - u_true_initial) ** 2))
        norm_i = max(float(np.mean(u_true_initial**2)), 1.0)
        rel_l2_i = np.sqrt(mse_i) / np.sqrt(norm_i)
        return float((rel_l2_f + rel_l2_b + rel_l2_i) / 3.0)

    return float((rel_l2_f + rel_l2_b) / 2.0)


def evaluate_expression(
    expr_str: str,
    pde: BasePDE,
    n_points: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    symbols = {name: sp.Symbol(name) for name in pde.get_variable_names()}
    try:
        expr = sp.sympify(expr_str, locals=symbols)
    except Exception as exc:
        return {"l_phy": float("inf"), "pre": False, "mse": float("inf"), "relative_mse": float("inf"), "error": str(exc)}

    try:
        expr_func = sp.lambdify(list(symbols.values()), expr, modules=["numpy"])
        x_test = pde.sample_domain(n_points, seed=seed)
        args = [x_test[:, i] for i in range(x_test.shape[1])]
        u_pred = expr_func(*args)
        if isinstance(u_pred, (int, float)):
            u_pred = np.full(n_points, u_pred)
        u_pred = np.asarray(u_pred, dtype=float)
        if not np.all(np.isfinite(u_pred)):
            raise ValueError("Non-finite expression evaluation")
        u_true = pde.ground_truth(x_test).flatten()
        mse = float(np.mean((u_pred - u_true) ** 2))
        denom = float(np.sum(u_true**2))
        relative_mse = float(np.sum((u_pred - u_true) ** 2) / denom) if denom > 0 else float("inf")
    except Exception:
        mse = float("inf")
        relative_mse = float("inf")

    return {
        "l_phy": compute_l_phy(expr, pde, n_domain=n_points, n_boundary=max(100, n_points // 2), seed=seed),
        "pre": compute_pre(expr, pde, n_test=n_points, seed=seed),
        "mse": mse,
        "mse_vs_gt": mse,
        "relative_mse": relative_mse,
        "expression": expr_str,
    }
