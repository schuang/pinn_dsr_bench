from .advection2d_cases import Advection2DGaussianCase
from .poisson2d_cases import Poisson2DPolynomialCase


CASE_REGISTRY = {
    "poisson2d_polynomial": Poisson2DPolynomialCase,
    "advection2d_gaussian": Advection2DGaussianCase,
}


def get_case(case_id: str):
    try:
        return CASE_REGISTRY[case_id]()
    except KeyError as exc:
        raise ValueError(f"Unknown case: {case_id}") from exc


def list_cases():
    return sorted(CASE_REGISTRY)
