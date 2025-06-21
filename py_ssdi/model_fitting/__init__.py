# py_ssdi/model_fitting/__init__.py
"""
Model–fitting utilities.

Exposes the most commonly used functions at package import:
    select_order, fit_var      – VAR model fitting
    (fit_ss, etc.)             – state-space fitting   ← add when implemented
"""

from .VAR_fitting import select_order, fit_var
# from .SS_fitting  import fit_ss   # ← once you create SS_fitting.py

__all__ = [
    "select_order",
    "fit_var",
    # "fit_ss",                    # add later
]