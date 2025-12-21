import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Any, List, Optional


DEFAULT_ROW_HEIGHT_PX = 18
DEFAULT_BASE_HEIGHT_PX = 120


def compute_uniform_axes(
    df: pd.DataFrame,
    row_key_cols: List[str],
    col_key_cols: List[str],
    value_col: str,
) -> tuple[Optional[tuple[float, float]], int]:
    """
    Compute global x-axis range and max row count for uniform charts.
    """
    if df.empty or not value_col or value_col not in df.columns:
        return None, 0

    plot_df = df.copy()
    key_cols = [c for c in (row_key_cols + col_key_cols) if c in plot_df.columns]
    for col_name in key_cols:
        plot_df = plot_df[plot_df[col_name].notna()].copy()
        plot_df[col_name] = plot_df[col_name].astype(str)

    plot_df["_val_num"] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna(subset=["_val_num"])

    if plot_df.empty:
        return None, 0

    x_max = float(plot_df["_val_num"].max())
    x_min = float(plot_df["_val_num"].min())

    if key_cols:
        counts = plot_df.groupby(key_cols)["_val_num"].size()
        y_max_count = int(counts.max()) if not counts.empty else 0
    else:
        y_max_count = int(len(plot_df))

    if x_min >= 0:
        x_min = 0.0

    if x_max == x_min:
        pad = 1.0 if x_max == 0 else abs(x_max) * 0.05
        x_min -= pad
        x_max += pad

    return (x_min, x_max), y_max_count


def build_uniform_spaghetti_fig(
    df: pd.DataFrame,
    subj_col: str,
    value_col: str,
    title: str,
    x_range: Optional[tuple[float, float]] = None,
    y_max_count: Optional[int] = None,
    agg_funcs: Optional[List[Any]] = None,
    agg_names: Optional[List[str]] = None,
    marker_color: Optional[str] = None,
    row_height_px: int = DEFAULT_ROW_HEIGHT_PX,
    base_height_px: int = DEFAULT_BASE_HEIGHT_PX,
) -> Optional["go.Figure"]:
    """
    Build a horizontal bar chart with a uniform axis scale and height.
    """
    if df.empty:
        return None

    tmp = df[[subj_col, value_col]].copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[value_col])

    if tmp.empty:
        return None

    tmp = tmp.sort_values(by=value_col, ascending=False)

    x_vals = tmp[value_col].values.tolist()
    y_labels = tmp[subj_col].values.tolist()

    if y_max_count is None:
        y_max_count = len(x_vals)
    if y_max_count <= 0:
        return None

    start_pos = y_max_count - 1
    y_positions = list(range(start_pos, start_pos - len(x_vals), -1))

    fig_height = base_height_px + y_max_count * row_height_px

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=y_positions,
            orientation="h",
            marker=dict(
                color=marker_color if marker_color else "#636efa",
                opacity=0.8,
            ),
            text=[f"{v:.2f}" for v in x_vals],
            textposition="outside",
            customdata=y_labels,
            hovertemplate=(
                f"<b>{subj_col}</b>: %{{customdata}}<br>"
                f"<b>{value_col}</b>: %{{x}}<br>"
                "<extra></extra>"
            ),
        )
    )

    if agg_funcs:
        for idx, func in enumerate(agg_funcs[:2]):
            if not callable(func):
                continue
            try:
                agg_value = func(pd.Series(x_vals))
                agg_x = float(agg_value)
            except Exception:
                continue

            agg_label = (
                agg_names[idx]
                if agg_names and idx < len(agg_names)
                else f"Agg {idx + 1}"
            )
            fig.add_vline(
                x=agg_x,
                line_width=3,
                line_dash="dash" if idx == 0 else "solid",
                line_color="red",
                annotation_text=f"{agg_label}: {agg_x:.2f}",
                annotation_position="top",
            )

    fig.update_layout(
        title=title,
        xaxis_title=value_col,
        yaxis_title=subj_col,
        height=fig_height,
        margin=dict(l=20, r=20, t=40, b=20),
        hoverlabel=dict(font=dict(size=16)),
    )

    yaxis_cfg = dict(
        range=[-0.5, y_max_count - 0.5],
        tickmode="array",
        tickvals=y_positions,
        ticktext=y_labels,
        type="linear",
    )
    fig.update_yaxes(**yaxis_cfg)

    if x_range:
        fig.update_xaxes(range=list(x_range))

    return fig


def render_uniform_spaghetti_fig(fig: "go.Figure", key: str) -> None:
    """
    Render chart and handle point selection.
    """
    st.plotly_chart(
        fig,
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key=key,
    )

    chart_state = st.session_state.get(key)
    if chart_state:
        selection = (
            chart_state.get("selection")
            if isinstance(chart_state, dict)
            else getattr(chart_state, "selection", None)
        )

        if selection and selection.get("points"):
            pt = selection["points"][0]
            custom_data = pt.get("customdata")
            if isinstance(custom_data, list):
                selected_id = custom_data[0]
            else:
                selected_id = custom_data

            if selected_id is None:
                selected_id = pt.get("y")

            if selected_id is not None:
                st.session_state["selected_subject_id"] = selected_id


def draw_uniform_spaghetti_chart(
    df: pd.DataFrame,
    subj_col: str,
    value_col: str,
    title: str,
    key: str,
    x_range: Optional[tuple[float, float]] = None,
    y_max_count: Optional[int] = None,
    agg_funcs: Optional[List[Any]] = None,
    agg_names: Optional[List[str]] = None,
    marker_color: Optional[str] = None,
) -> None:
    fig = build_uniform_spaghetti_fig(
        df=df,
        subj_col=subj_col,
        value_col=value_col,
        title=title,
        x_range=x_range,
        y_max_count=y_max_count,
        agg_funcs=agg_funcs,
        agg_names=agg_names,
        marker_color=marker_color,
    )

    if fig is None:
        st.info("No valid numeric data for this chart.")
        return

    render_uniform_spaghetti_fig(fig, key=key)
