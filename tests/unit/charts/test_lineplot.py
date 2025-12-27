"""Tests for line plot helpers."""

import pandas as pd

from analysis.plugins.charts.lineplot import build_pivot_line_fig


def _sample_df() -> pd.DataFrame:
    """Return a minimal DataFrame for lineplot tests."""
    return pd.DataFrame(
        {
            "Group": ["A", "A", "B", "B"],
            "Visit": ["V1", "V2", "V1", "V2"],
            "Value": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_lineplot_error_bars_and_counts() -> None:
    """Ensure error bars and count labels are emitted when enabled."""
    df = _sample_df()
    fig = build_pivot_line_fig(
        df=df,
        value_col="Value",
        row_key_cols=["Group"],
        col_field="Visit",
        agg_name="Mean - 平均值",
        error_mode="SE",
        show_counts=True,
    )
    assert fig is not None
    assert len(fig.data) == 2
    assert fig.data[0].error_y is not None
    ticktext = fig.layout.xaxis.ticktext
    assert ticktext is not None
    assert any("n=" in str(text) for text in ticktext)


def test_lineplot_without_counts() -> None:
    """Ensure count labels are omitted when disabled."""
    df = _sample_df()
    fig = build_pivot_line_fig(
        df=df,
        value_col="Value",
        row_key_cols=["Group"],
        col_field="Visit",
        agg_name="Median - 中位数",
    )
    assert fig is not None
    assert fig.layout.xaxis.ticktext is None
