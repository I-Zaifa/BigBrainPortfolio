from ui import plots


def test_allocation_pie(tmp_path):
    weights = {"A": 0.5, "B": 0.5}
    # Plotting shouldn't raise errors; Streamlit's plot function is mocked.
    class Dummy:
        def plotly_chart(self, *a, **k):
            pass
    original = plots.st
    plots.st = Dummy()
    try:
        plots.allocation_pie(weights)
    finally:
        plots.st = original
