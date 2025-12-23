from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis_methods import AGG_METHODS


def _coerce_number(val: Any) -> Optional[float]:
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
    if not row_key_cols:
        return "All"
    values = [str(row_key.get(col, "")) for col in row_key_cols]
    return " / ".join(values)


def _format_row_hover(
    row_key: Dict[str, str], row_key_cols: List[str]
) -> str:
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
) -> Optional["go.Figure"]:
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

    agg_func = AGG_METHODS.get(agg_name, "mean")
    group_cols = list(row_key_cols) + [col_field]
    grouped = work_df.groupby(group_cols, dropna=False, sort=False)
    agg_map: Dict[Tuple[Tuple[str, ...], str], Optional[float]] = {}
    for key, group in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        row_tuple = tuple(key_tuple[: len(row_key_cols)])
        col_val = key_tuple[len(row_key_cols)]
        series = group[value_col]
        try:
            if callable(agg_func):
                agg_val = agg_func(series)
            else:
                agg_val = series.agg(agg_func)
        except Exception:
            agg_val = None
        agg_map[(row_tuple, str(col_val))] = _coerce_number(agg_val)

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
            )
        )

    if not fig.data:
        return None

    fig.update_layout(
        height=360,
        margin=dict(l=40, r=20, t=30, b=90),
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
    fig.update_xaxes(
        title=col_field,
        type="category",
        categoryorder="array",
        categoryarray=x_values,
        automargin=True,
    )
    fig.update_yaxes(
        title=f"{agg_name} ({value_col})",
        automargin=True,
        showgrid=True,
        gridcolor="#e6e6e6",
    )
    return fig


def render_line_fig(fig: "go.Figure", key: str) -> None:
    st.plotly_chart(fig, width="stretch", key=key)
