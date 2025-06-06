# BigBrainPortfolio

This project provides a simple **Smart Portfolio Allocator** using Modern Portfolio Theory.
It allows users to construct an optimized stock portfolio based on risk tolerance,
sector preferences, and ESG considerations.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Run the allocator from the command line. Example:

```bash
python portfolio_allocator.py --amount 10000 --risk High --sectors "Technology,Healthcare" --esg --nstocks 8
```

The program downloads historical prices for a small universe of stocks
(from `tickers.csv`), optimizes the portfolio, and outputs allocations to
`portfolio_allocation.csv` along with performance metrics.

## Data

`tickers.csv` includes example tickers with sector information and mocked ESG scores.
Real deployments should replace this with up‑to‑date data.

## Streamlit Web App

A simple web interface is provided via Streamlit. Launch it with:

```bash
streamlit run streamlit_app.py
```

The app lets you pick sectors, set risk preferences and visualize the optimized
portfolio with interactive charts.

