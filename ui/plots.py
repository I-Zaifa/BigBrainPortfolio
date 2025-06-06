"""Plotting utilities for the Streamlit app.

The functions defined here encapsulate our Plotly chart generation so
that the main app logic remains clean. By isolating plotting code we can
also more easily test that charts are produced without error.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

__all__ = ["allocation_pie", "efficient_frontier_plot"]


def allocation_pie(weights: dict) -> None:
    """Render a donut chart of portfolio weights."""
    fig = go.Figure(
        go.Pie(labels=list(weights.keys()), values=list(weights.values()), hole=0.45)
    )
    st.plotly_chart(fig, use_container_width=True)


def efficient_frontier_plot(ef) -> None:
    """Plot the efficient frontier and current portfolio location."""
    w_min, ret_min, vol_min = ef.portfolio_performance()
    frontier = ef.efficient_frontier(points=100)
    xs, ys = zip(*[(vol, ret) for ret, vol in frontier])
    fig = go.Figure()
    fig.add_scatter(x=xs, y=ys, mode="lines", name="Frontier")
    fig.add_scatter(
        x=[vol_min],
        y=[ret_min],
        mode="markers",
        name="Your Portfolio",
        marker_size=[10],
    )
    st.plotly_chart(fig, use_container_width=True)

