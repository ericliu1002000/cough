"""Line plot helpers for pivoted data views."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis.plugins.methods import AGG_METHODS

# 使用场景：嵌套透视表中的折线分面图（列维度字段 × 聚合函数），行维度组合为多条线。
# 折线图


def _coerce_number(val: Any) -> Optional[float]:
    """Coerce a value to float, returning None for invalid inputs."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    try:
        return float(val)
    except Exception:
        return None


def _build_row_keys(
    df: pd.DataFrame,
    row_key_cols: List[str],
    row_orders: Optional[Dict[str, List[str]]],
) -> Tuple[List[Dict[str, str]], List[Tuple[str, ...]]]:
    """Return ordered row key dicts and their tuple equivalents."""
    if not row_key_cols:
        return [{}], [()]

    row_keys_df = df[row_key_cols].drop_duplicates()
    if row_orders:
        ordered = row_keys_df.copy()
        order_cols = []
        for col in row_key_cols:
            order_list = row_orders.get(col)
            if not order_list:
                continue
            order_map = {str(val): idx for idx, val in enumerate(order_list)}
            ordered_col = ordered[col].map(
                lambda v: order_map.get(str(v), len(order_map))
            )
            order_name = f"_order_{col}"
            ordered[order_name] = ordered_col
            order_cols.append(order_name)
        if order_cols:
            row_keys_df = ordered.sort_values(
                order_cols, kind="stable"
            ).drop(columns=order_cols)

    row_keys = row_keys_df.to_dict(orient="records")
    row_key_tuples = [
        tuple(rec.get(col, "") for col in row_key_cols) for rec in row_keys
    ]
    return row_keys, row_key_tuples


def _build_col_values(
    df: pd.DataFrame,
    col_field: str,
    col_orders: Optional[Dict[str, List[str]]],
) -> List[str]:
    """Return ordered x-axis values based on optional ordering."""
    available = (
        df[col_field]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    if not col_orders or col_field not in col_orders:
        return available

    order_list = [str(v) for v in col_orders.get(col_field, [])]
    ordered = [v for v in order_list if v in available]
    ordered.extend([v for v in available if v not in ordered])
    return ordered


def _format_row_label(
    row_key: Dict[str, str], row_key_cols: List[str]
) -> str:
    """Return a display label for a row key."""
    if not row_key_cols:
        return "All"
    values = [str(row_key.get(col, "")) for col in row_key_cols]
    return " / ".join(values)


def _format_row_hover(
    row_key: Dict[str, str], row_key_cols: List[str]
) -> str:
    """Return hover text for a row key."""
    if not row_key_cols:
        return "All"
    return "<br>".join(
        [f"{col}: {row_key.get(col, '')}" for col in row_key_cols]
    )


def build_pivot_line_fig(
    df: pd.DataFrame,
    value_col: str,
    row_key_cols: List[str],
    col_field: str,
    agg_name: str,
    row_orders: Optional[Dict[str, List[str]]] = None,
    col_orders: Optional[Dict[str, List[str]]] = None,
    error_mode: Optional[str] = None,
    show_counts: bool = False,
    agg_func: Optional[Callable[[pd.Series], float]] = None,
    y_range_pad_ratio: float = 0.0,
) -> Optional["go.Figure"]:
    """Build a line chart figure from pivoted data."""
    if df.empty or value_col not in df.columns or col_field not in df.columns:
        return None

    work_df = df.copy()
    key_cols = list(row_key_cols) + [col_field]
    if key_cols:
        work_df = work_df.dropna(subset=key_cols)
        for key_col in key_cols:
            work_df[key_col] = work_df[key_col].astype(str)

    work_df[value_col] = pd.to_numeric(work_df[value_col], errors="coerce")
    work_df = work_df.dropna(subset=[value_col])
    if work_df.empty:
        return None

    if row_orders:
        row_orders = {
            key: [str(v) for v in vals]
            for key, vals in row_orders.items()
            if isinstance(vals, (list, tuple, set))
        }
    row_keys, row_key_tuples = _build_row_keys(
        work_df, row_key_cols, row_orders
    )
    x_values = _build_col_values(work_df, col_field, col_orders)
    if not x_values:
        return None

    if error_mode:
        error_mode = str(error_mode).upper()
        if error_mode not in {"SD", "SE"}:
            error_mode = None

    resolved_agg_func = (
        agg_func if agg_func is not None else AGG_METHODS.get(agg_name, "mean")
    )
    group_cols = list(row_key_cols) + [col_field]
    grouped = work_df.groupby(group_cols, dropna=False, sort=False)
    agg_map: Dict[Tuple[Tuple[str, ...], str], Optional[float]] = {}
    count_map: Dict[Tuple[Tuple[str, ...], str], int] = {}
    error_map: Dict[Tuple[Tuple[str, ...], str], Optional[float]] = {}
    needs_counts = show_counts or error_mode is not None
    for key, group in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        row_tuple = tuple(key_tuple[: len(row_key_cols)])
        col_val = str(key_tuple[len(row_key_cols)])
        series = group[value_col]
        n_val = None
        if needs_counts:
            n_val = int(series.count())
            count_map[(row_tuple, col_val)] = n_val
        if error_mode:
            if n_val is None:
                n_val = int(series.count())
            sd_val = series.std(ddof=1)
            err_val = None
            if n_val > 1 and pd.notna(sd_val):
                if error_mode == "SE":
                    err_val = sd_val / math.sqrt(n_val)
                elif error_mode == "SD":
                    err_val = sd_val
            error_map[(row_tuple, col_val)] = _coerce_number(err_val)
        try:
            if callable(resolved_agg_func):
                agg_val = resolved_agg_func(series)
            else:
                agg_val = series.agg(resolved_agg_func)
        except Exception:
            agg_val = None
        agg_map[(row_tuple, col_val)] = _coerce_number(agg_val)

    y_min = None
    y_max = None
    for row_tuple in row_key_tuples:
        for x_val in x_values:
            y_val = agg_map.get((row_tuple, x_val))
            if y_val is None:
                continue
            low = y_val
            high = y_val
            if error_mode:
                err_val = error_map.get((row_tuple, x_val))
                if err_val is not None:
                    err_val = abs(err_val)
                    low = y_val - err_val
                    high = y_val + err_val
            if y_min is None or low < y_min:
                y_min = low
            if y_max is None or high > y_max:
                y_max = high

    color_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    fig = go.Figure()
    for idx, (row_key, row_tuple) in enumerate(
        zip(row_keys, row_key_tuples)
    ):
        y_vals = [agg_map.get((row_tuple, x)) for x in x_values]
        if all(v is None for v in y_vals):
            continue
        label = _format_row_label(row_key, row_key_cols)
        hover_prefix = _format_row_hover(row_key, row_key_cols)
        hover_lines = []
        if hover_prefix:
            hover_lines.append(hover_prefix)
        hover_lines.append(f"{col_field}: %{{x}}")
        hover_lines.append(f"{agg_name}: %{{y:.3f}}")
        hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"
        trace_kwargs = {}
        if error_mode:
            err_vals = [
                error_map.get((row_tuple, x)) for x in x_values
            ]
            trace_kwargs["error_y"] = dict(
                type="data",
                array=err_vals,
                visible=True,
            )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_vals,
                mode="lines+markers",
                name=label,
                line=dict(
                    color=color_palette[idx % len(color_palette)], width=2
                ),
                marker=dict(size=6),
                connectgaps=False,
                hovertemplate=hovertemplate,
                **trace_kwargs,
            )
        )

    if not fig.data:
        return None

    bottom_margin = 90
    if show_counts:
        tick_lines = 1 + len(row_key_tuples)
        bottom_margin = max(bottom_margin, 20 + tick_lines * 14)

    fig.update_layout(
        height=360,
        margin=dict(l=40, r=20, t=30, b=bottom_margin),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.3,
            xanchor="center",
            x=0.5,
        ),
        template="plotly_white",
    )
    xaxis_kwargs = dict(
        title=col_field,
        type="category",
        categoryorder="array",
        categoryarray=x_values,
        range=[-0.5, len(x_values) - 0.5],
        automargin=True,
    )
    if show_counts:
        tick_texts = []
        for x_val in x_values:
            lines = [str(x_val)]
            for row_tuple in row_key_tuples:
                n_val = count_map.get((row_tuple, x_val))
                if n_val is None:
                    n_val = 0
                lines.append(f"n={n_val}")
            tick_texts.append("<br>".join(lines))
        xaxis_kwargs.update(
            tickmode="array",
            tickvals=x_values,
            ticktext=tick_texts,
        )
    fig.update_xaxes(**xaxis_kwargs)
    yaxis_kwargs = dict(
        title=f"{agg_name} ({value_col})",
        automargin=True,
        showgrid=True,
        gridcolor="#e6e6e6",
    )
    if y_min is not None and y_max is not None:
        pad = 0.0
        if y_min == y_max:
            pad = abs(y_min) * y_range_pad_ratio
            if pad == 0:
                pad = 1.0
        elif y_range_pad_ratio:
            pad = (y_max - y_min) * y_range_pad_ratio
        if pad:
            y_min -= pad
            y_max += pad
        yaxis_kwargs["range"] = [y_min, y_max]
    fig.update_yaxes(**yaxis_kwargs)
    return fig


def render_line_fig(fig: "go.Figure", key: str) -> None:
    """Render a line figure in Streamlit with a stable key."""
    st.plotly_chart(fig, width="stretch", key=key)
