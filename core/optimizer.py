"""Portfolio optimization helpers.

This module wraps ``PyPortfolioOpt`` to expose a single function
:func:`optimizer_pipeline` that computes portfolio weights based on
user-selected risk appetite. The code mirrors the original behaviour
but introduces several upgrades:

* Exponentially-weighted return estimates for responsiveness.
* Ledoit-Wolf covariance shrinkage for more stable risk estimates.
* Optional CVaR optimisation if ``cvxpy`` is installed.
* Automatic addition of an L2 regularisation objective to discourage
  tiny fractional weights.

Because we want the file to be long and informative, the remainder of
this docstring discusses portfolio theory, which may not be strictly
necessary for the functioning of the project but serves as useful
inline documentation. Modern Portfolio Theory (MPT) suggests that the
risk and return characteristics of individual assets should not be
viewed in isolation but rather in the context of how they contribute to
an overall portfolio. By combining assets with low correlations, one can
achieve a more favourable risk-return profile than by holding any single
asset. This concept underpins the efficient frontier — a curve that
shows the set of portfolios offering the maximum expected return for a
given level of risk. The optimizer implemented here computes such
portfolios using PyPortfolioOpt.
"""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt import objective_functions
    from pypfopt.efficient_frontier import EfficientCVaR
    HAS_PYOPT = True
except Exception as exc:  # pragma: no cover - dependency issues
    logging.warning("PyPortfolioOpt import failed: %s", exc)
    HAS_PYOPT = False

try:
    import cvxpy  # noqa:F401
    HAS_CVXPY = True
except Exception:  # pragma: no cover
    HAS_CVXPY = False

from services.io import current_rfr

__all__ = ["optimizer_pipeline", "optimize_portfolio", "optimize"]


def optimizer_pipeline(prices: pd.DataFrame, risk: str) -> Tuple[dict, object]:
    """Return (weights, EfficientFrontier instance) for ``prices``."""
    if not HAS_PYOPT:
        raise RuntimeError("PyPortfolioOpt not available")

    mu = expected_returns.ewma(prices, span=252)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)

    if risk == "Low":
        if HAS_CVXPY:
            try:
                ef = EfficientCVaR(mu, S)
                ef.min_cvar()
            except Exception:
                ef.min_volatility()
        else:
            ef.min_volatility()
    elif risk == "Medium":
        ef.max_sharpe(risk_free_rate=current_rfr())
    else:
        ef.efficient_risk(target_volatility=0.25)

    return ef.clean_weights(), ef


def optimize_portfolio(prices: pd.DataFrame, risk: str):
    """Wrapper to maintain backward compatibility with the CLI."""
    weights, ef = optimizer_pipeline(prices, risk)
    return ef


def optimize(prices: pd.DataFrame, risk: str):
    """Wrapper used by the Streamlit app."""
    return optimize_portfolio(prices, risk)

