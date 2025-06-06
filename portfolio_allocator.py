import argparse
import sys
from typing import List, Optional

import pandas as pd
import yfinance as yf
from pypfopt import EfficientFrontier, expected_returns, risk_models, DiscreteAllocation
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Portfolio Allocator")
    parser.add_argument("--amount", type=float, required=True, help="Investment amount in USD")
    parser.add_argument("--risk", choices=["Low", "Medium", "High"], required=True, help="Risk tolerance")
    parser.add_argument("--sectors", type=str, default="", help="Comma separated sector preferences")
    parser.add_argument("--esg", action="store_true", help="Filter for ESG score > 70")
    parser.add_argument("--nstocks", type=int, default=10, help="Number of stocks to include")
    parser.add_argument("--output", type=str, default="portfolio_allocation.csv", help="CSV output file")
    return parser.parse_args()


def load_tickers(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def filter_tickers(df: pd.DataFrame, sectors: Optional[List[str]], esg: bool) -> pd.DataFrame:
    if sectors:
        df = df[df["Sector"].isin(sectors)]
    if esg:
        df = df[df["ESGScore"] >= 70]
    return df


def fetch_price_data(tickers: List[str]) -> pd.DataFrame:
    data = yf.download(tickers, period="1y", interval="1d", progress=False)["Adj Close"]
    return data.dropna(axis=1, how="any")


def optimize_portfolio(prices: pd.DataFrame, risk: str) -> EfficientFrontier:
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu, S)
    if risk == "High":
        ef.max_sharpe()
    elif risk == "Medium":
        target_volatility = 0.2
        ef.efficient_risk(target_volatility)
    else:
        ef.min_volatility()
    return ef


def allocate_discrete(ef: EfficientFrontier, prices: pd.DataFrame, amount: float):
    weights = ef.clean_weights()
    latest_prices = prices.iloc[-1]
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=amount)
    allocation, leftover = da.lp_portfolio()
    perf = ef.portfolio_performance()
    return allocation, leftover, perf, weights


def main():
    args = parse_args()
    sectors = [s.strip() for s in args.sectors.split(",") if s.strip()] if args.sectors else None
    tickers_df = load_tickers("tickers.csv")
    filtered = filter_tickers(tickers_df, sectors, args.esg)
    if filtered.empty:
        print("No tickers available after filtering.")
        sys.exit(1)
    if len(filtered) < args.nstocks:
        args.nstocks = len(filtered)
    universe = filtered.sample(n=args.nstocks, random_state=42)
    prices = fetch_price_data(universe["Ticker"].tolist())
    ef = optimize_portfolio(prices, args.risk)
    allocation, leftover, perf, weights = allocate_discrete(ef, prices, args.amount)

    alloc_df = pd.DataFrame({
        "Ticker": list(weights.keys()),
        "Weight": list(weights.values())
    })
    alloc_df = alloc_df[alloc_df["Weight"] > 0]
    alloc_df["Shares"] = alloc_df["Ticker"].map(allocation)
    alloc_df.to_csv(args.output, index=False)

    exp_ret, vol, sharpe = perf
    print("Allocation saved to", args.output)
    print("Expected annual return: {:.2%}".format(exp_ret))
    print("Annual volatility: {:.2%}".format(vol))
    print("Sharpe Ratio: {:.2f}".format(sharpe))
    print("Leftover cash: ${:.2f}".format(leftover))

    # Plot Efficient Frontier
    try:
        from pypfopt import plotting
        fig, ax = plt.subplots()
        plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
        plt.scatter(vol, exp_ret, marker="*", color="r", s=100, label="Optimized Portfolio")
        plt.title("Efficient Frontier")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Could not plot efficient frontier:", e)


if __name__ == "__main__":
    main()
