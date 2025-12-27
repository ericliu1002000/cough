"""Plugin method registry and public exports."""

from typing import Any, Callable, Dict

CALC_METHODS: Dict[str, Callable] = {}
AGG_METHODS: Dict[str, Any] = {}


def register_calc_method(name: str):
    """Return a decorator that registers a row-wise calculation."""

    def decorator(func):
        """Register the calculation function in CALC_METHODS."""
        CALC_METHODS[name] = func
        return func

    return decorator


def register_agg_method(name: str):
    """Return a decorator that registers an aggregation method."""

    def decorator(func):
        """Register the aggregation function in AGG_METHODS."""
        AGG_METHODS[name] = func
        return func

    return decorator


# Import modules so decorators are executed and registries are populated.
from . import agg as _agg  # noqa: F401
from . import calc as _calc  # noqa: F401
from . import composite as _composite  # noqa: F401

# Re-export commonly used helpers for backwards compatibility.
from .agg import sas_quantile
from .composite import (
    calculate_anova_f_test,
    calculate_t_test_from_summary,
    calculate_proportion_p_value,
)

__all__ = [
    "AGG_METHODS",
    "CALC_METHODS",
    "register_agg_method",
    "register_calc_method",
    "sas_quantile",
    "calculate_anova_f_test",
    "calculate_t_test_from_summary",
    "calculate_proportion_p_value",
]
