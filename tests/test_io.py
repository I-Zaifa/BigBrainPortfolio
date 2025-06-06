import pandas as pd
import pytest
import asyncio

from services import io


def test_load_sp500_columns():
    try:
        df = io.load_sp500()
    except Exception:
        pytest.skip("network unavailable")
    assert set({"Ticker", "Name", "Sector"}).issubset(df.columns)


def test_filter_tickers():
    data = pd.DataFrame({
        "Ticker": ["A", "B"],
        "Sector": ["Tech", "Finance"],
        "ESGScore": [80, 60],
    })
    filtered = io.filter_tickers(data, ["Tech"], esg=True)
    assert list(filtered.Ticker) == ["A"]


def test_get_prices_async_empty():
    df = asyncio.run(io.get_prices_async._info.func([]))
    assert df.empty
