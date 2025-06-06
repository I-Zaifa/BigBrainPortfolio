import pandas as pd
import pytest

from core import optimizer


def test_optimizer_pipeline_simple():
    prices = pd.DataFrame({
        "A": [1, 1.1, 1.2, 1.15],
        "B": [1, 0.9, 1.05, 1.1],
    })
    try:
        weights, ef = optimizer.optimizer_pipeline(prices, "Medium")
    except AttributeError:
        pytest.skip("Incompatible PyPortfolioOpt")
    assert isinstance(weights, dict)
    assert ef is not None
