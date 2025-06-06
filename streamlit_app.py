import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, expected_returns, risk_models
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


def load_tickers(path: str = "tickers.csv") -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def fetch_prices(tickers: List[str]) -> pd.DataFrame:
    data = yf.download(tickers, period="1y", interval="1d", progress=False)["Adj Close"]
    return data.ffill().dropna(how="all")


def filter_universe(df: pd.DataFrame, sectors: Optional[List[str]], esg: bool) -> pd.DataFrame:
    if sectors:
        df = df[df["Sector"].isin(sectors)]
    if esg:
        df = df[df["ESGScore"] >= 70]
    return df


def optimize(prices: pd.DataFrame, risk: str) -> EfficientFrontier:
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu, S)
    if risk == "High":
        ef.max_sharpe()
    elif risk == "Medium":
        ef.efficient_risk(0.2)
    else:
        ef.min_volatility()
    return ef


def portfolio_cumulative_returns(prices: pd.DataFrame, weights: dict) -> pd.Series:
    weights_series = pd.Series(weights)
    weights_series = weights_series[weights_series > 0]
    weighted = prices[weights_series.index] * weights_series
    portfolio = weighted.sum(axis=1)
    returns = portfolio.pct_change().fillna(0)
    return (1 + returns).cumprod()


def sector_exposure(weights: dict, meta: pd.DataFrame) -> pd.Series:
    df = pd.DataFrame({"Ticker": list(weights.keys()), "Weight": list(weights.values())})
    df = df.merge(meta, left_on="Ticker", right_on="Ticker", how="left")
    exposure = df.groupby("Sector")["Weight"].sum()
    return exposure


def main():
    st.title("Smart Portfolio Allocator")

    companies = load_tickers()
    all_sectors = sorted(companies["Sector"].unique())

    # Sidebar inputs
    amount = st.sidebar.number_input("Investment amount ($)", min_value=1000.0, value=10000.0, step=1000.0)
    risk = st.sidebar.selectbox("Risk tolerance", ["Low", "Medium", "High"], index=2)
    selected_sectors = st.sidebar.multiselect("Sectors", all_sectors)
    esg_filter = st.sidebar.checkbox("ESG score >= 70")
    nstocks = st.sidebar.slider("Number of stocks", min_value=2, max_value=20, value=8)

    filtered = filter_universe(companies, selected_sectors, esg_filter)

    st.subheader("Candidate Stocks")
    st.dataframe(filtered)

    if st.button("Optimize Portfolio"):
        if filtered.empty:
            st.warning("No tickers available with the current filters.")
            return
        if len(filtered) < nstocks:
            nstocks = len(filtered)
        universe = filtered.sample(n=nstocks, random_state=42)
        prices = fetch_prices(universe["Ticker"].tolist())
        ef = optimize(prices, risk)
        weights = ef.clean_weights()
        perf = ef.portfolio_performance()
        cum_ret = portfolio_cumulative_returns(prices, weights)
        sp500 = fetch_prices(["^GSPC"]).squeeze()
        bench_cum = (1 + sp500.pct_change().fillna(0)).cumprod()
        exposure = sector_exposure(weights, companies)

        st.subheader("Portfolio Weights")
        weight_df = pd.DataFrame(list(weights.items()), columns=["Ticker", "Weight"])
        st.dataframe(weight_df[weight_df["Weight"] > 0])

        exp_ret, vol, sharpe = perf
        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Return", f"{exp_ret:.2%}")
        col2.metric("Volatility", f"{vol:.2%}")
        col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

        # Allocation pie chart
        fig_alloc = px.pie(weight_df[weight_df["Weight"] > 0], names="Ticker", values="Weight", title="Allocation")
        st.plotly_chart(fig_alloc, use_container_width=True)

        # Sector exposure bar
        fig_sector = px.bar(exposure, title="Sector Exposure")
        st.plotly_chart(fig_sector, use_container_width=True)

        # Cumulative returns line chart
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret, name="Portfolio"))
        fig_line.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum, name="S&P 500"))
        fig_line.update_layout(title="Cumulative Returns", xaxis_title="Date", yaxis_title="Growth of $1")
        st.plotly_chart(fig_line, use_container_width=True)

        # Efficient frontier (static via matplotlib converted to plotly)
        try:
            from pypfopt import plotting
            import matplotlib.pyplot as plt
            ef_points = plotting._efficient_frontier(ef, "return", None, 50)
            ef_x = ef_points["risk"]
            ef_y = ef_points["return"]
            fig_frontier = go.Figure()
            fig_frontier.add_trace(go.Scatter(x=ef_x, y=ef_y, mode="lines", name="Efficient Frontier"))
            fig_frontier.add_trace(go.Scatter(x=[vol], y=[exp_ret], mode="markers", marker=dict(color="red", size=10), name="Your Portfolio"))
            fig_frontier.update_layout(title="Efficient Frontier", xaxis_title="Volatility", yaxis_title="Return")
            st.plotly_chart(fig_frontier, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate efficient frontier: {e}")

        # Download weights
        csv = weight_df.to_csv(index=False).encode()
        st.download_button("Download Weights", csv, "weights.csv", "text/csv")


if __name__ == "__main__":
    main()
