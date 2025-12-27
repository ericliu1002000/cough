"""Boxplot helpers for pivoted data views."""

import html
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 使用场景：统一坐标系箱线图（行×列组合在同一坐标中），用于分布对比与异常点查看。


def compute_boxplot_range(
    df: pd.DataFrame, value_col: str
) -> Optional[tuple[float, float]]:
    """
    Compute a global Y-axis range so each cell uses the same scale.
    """
    if df.empty or not value_col or value_col not in df.columns:
        return None

    values = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if values.empty:
        return None

    y_min = float(values.min())
    y_max = float(values.max())

    if y_min == y_max:
        pad = 1.0 if y_min == 0 else abs(y_min) * 0.05
        y_min -= pad
        y_max += pad

    return (y_min, y_max)


def build_boxplot_matrix_fig(
    df: pd.DataFrame,
    subj_col: str,
    value_col: str,
    row_key_cols: List[str],
    col_key_cols: List[str],
    row_keys: Optional[List[Dict[str, str]]] = None,
    col_keys: Optional[List[Dict[str, str]]] = None,
    combo_keys: Optional[List[Dict[str, str]]] = None,
    y_range: Optional[tuple[float, float]] = None,
    color_labels: Optional[List[str]] = None,
) -> Optional["go.Figure"]:
    """
    Build a single boxplot figure with one global coordinate system.
    Each row/column combination becomes a category on the X axis.
    """
    if df.empty or value_col not in df.columns:
        return None

    tmp = df.copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[value_col])
    if tmp.empty:
        return None

    local_row_cols = [c for c in row_key_cols if c in tmp.columns]
    local_col_cols = [c for c in col_key_cols if c in tmp.columns]
    if not local_row_cols and local_col_cols:
        local_row_cols = list(local_col_cols)
        local_col_cols = []
        if row_keys is None and col_keys is not None:
            row_keys = col_keys
        col_keys = [{}]

    key_cols = local_row_cols + local_col_cols
    for col_name in key_cols:
        tmp = tmp[tmp[col_name].notna()].copy()
        tmp[col_name] = tmp[col_name].astype(str)

    if row_keys is None:
        if local_row_cols:
            row_keys = (
                tmp[local_row_cols]
                .drop_duplicates()
                .to_dict(orient="records")
            )
        else:
            row_keys = [{}]

    if col_keys is None:
        if local_col_cols:
            col_keys = (
                tmp[local_col_cols]
                .drop_duplicates()
                .to_dict(orient="records")
            )
        else:
            col_keys = [{}]

    if not row_keys or not col_keys:
        return None

    def build_key(record: Dict[str, str], cols: List[str]) -> str:
        """Return a stable key string for row/column combinations."""
        if not cols:
            return "(All)"
        return "\x1f".join([str(record.get(col, "")) for col in cols])

    def build_axis_label(record: Dict[str, str], cols: List[str]) -> str:
        """Return a display label for x-axis groupings."""
        if not cols:
            return "(All)"
        return "<br>".join(
            [html.escape(str(record.get(col, ""))) for col in cols]
        )

    def build_legend_label(record: Dict[str, str], cols: List[str]) -> str:
        """Return a display label for legend groupings."""
        if not cols:
            return "All"
        return " / ".join(
            [html.escape(str(record.get(col, ""))) for col in cols]
        )

    allowed_pairs = None
    if combo_keys:
        allowed_pairs = set()
        for combo in combo_keys:
            row_part = {c: combo.get(c, "") for c in local_row_cols}
            col_part = {c: combo.get(c, "") for c in local_col_cols}
            allowed_pairs.add(
                (
                    build_key(row_part, local_row_cols),
                    build_key(col_part, local_col_cols),
                )
            )

    row_order_keys: list[str] = []
    row_label_map: dict[str, str] = {}
    for row_item in row_keys:
        key = build_key(row_item, local_row_cols)
        if key not in row_label_map:
            row_order_keys.append(key)
            row_label_map[key] = build_axis_label(row_item, local_row_cols)

    col_order_keys: list[str] = []
    col_label_map: dict[str, str] = {}
    for col_item in col_keys:
        key = build_key(col_item, local_col_cols)
        if key not in col_label_map:
            col_order_keys.append(key)
            col_label_map[key] = build_legend_label(col_item, local_col_cols)

    if allowed_pairs:
        allowed_rows = {row_key for row_key, _ in allowed_pairs}
        allowed_cols = {col_key for _, col_key in allowed_pairs}
        row_order_keys = [
            key for key in row_order_keys if key in allowed_rows
        ]
        col_order_keys = [
            key for key in col_order_keys if key in allowed_cols
        ]
        row_label_map = {
            key: row_label_map[key] for key in row_order_keys
        }
        col_label_map = {
            key: col_label_map[key] for key in col_order_keys
        }

    if local_row_cols:
        tmp["_row_key"] = tmp[local_row_cols].astype(str).agg(
            "\x1f".join, axis=1
        )
    else:
        tmp["_row_key"] = "(All)"

    if local_col_cols:
        tmp["_col_key"] = tmp[local_col_cols].astype(str).agg(
            "\x1f".join, axis=1
        )
    else:
        tmp["_col_key"] = "(All)"

    if allowed_pairs:
        allowed_pair_keys = {
            f"{row_key}\x1e{col_key}"
            for row_key, col_key in allowed_pairs
        }
        tmp["_pair_key"] = tmp["_row_key"] + "\x1e" + tmp["_col_key"]
        tmp = tmp[tmp["_pair_key"].isin(allowed_pair_keys)]
        if tmp.empty:
            return None

    custom_cols = []
    if subj_col and subj_col in tmp.columns:
        custom_cols.append(subj_col)
    custom_cols.extend(local_row_cols)
    custom_cols.extend(local_col_cols)

    color_palette = [
        "#636efa",
        "#ef553b",
        "#00cc96",
        "#ab63fa",
        "#ffa15a",
        "#19d3f3",
        "#ff6692",
        "#b6e880",
        "#ff97ff",
        "#fecb52",
    ]

    if color_labels is None:
        color_labels = [col_label_map[key] for key in col_order_keys]
    color_map = {
        label: color_palette[idx % len(color_palette)]
        for idx, label in enumerate(color_labels)
    }

    fig = go.Figure()
    for idx, col_key in enumerate(col_order_keys):
        group_df = tmp[tmp["_col_key"] == col_key]

        if group_df.empty:
            continue

        x_vals = [
            row_label_map.get(key, "")
            for key in group_df["_row_key"].tolist()
        ]
        y_vals = group_df[value_col].values.tolist()

        if custom_cols:
            customdata = (
                group_df[custom_cols].astype(str).values.tolist()
            )
        else:
            customdata = [None] * len(group_df)

        hover_parts: list[str] = []
        custom_offset = 0
        if subj_col and subj_col in tmp.columns:
            hover_parts.append(
                f"<b>{html.escape(subj_col)}</b>: %{{customdata[0]}}"
            )
            custom_offset = 1
        hover_parts.append(
            f"<b>{html.escape(value_col)}</b>: %{{y:.2f}}"
        )
        for idx, col_name in enumerate(local_row_cols + local_col_cols):
            hover_parts.append(
                f"<b>{html.escape(col_name)}</b>: "
                f"%{{customdata[{custom_offset + idx}]}}"
            )

        label = col_label_map.get(col_key, "All")
        color = color_map.get(label, color_palette[idx % len(color_palette)])

        fig.add_trace(
            go.Box(
                x=x_vals,
                y=y_vals,
                name=label,
                marker=dict(size=6, color=color),
                line=dict(color=color),
                boxpoints="outliers",
                jitter=0,
                pointpos=0,
                customdata=customdata,
                hoveron="points",
                hovertemplate="<br>".join(hover_parts) + "<extra></extra>",
                showlegend=True,
            )
        )

    if not fig.data:
        return None

    layout_kwargs = dict(
        xaxis_title="",
        yaxis_title=value_col,
        autosize=True,
        margin=dict(l=40, r=24, t=24, b=110),
        hoverlabel=dict(font=dict(size=14)),
        showlegend=len(col_order_keys) > 1,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            itemclick=False,
            itemdoubleclick=False,
        ),
        boxmode="group",
    )

    fig.update_layout(**layout_kwargs)

    if y_range:
        fig.update_yaxes(range=list(y_range))

    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=[row_label_map[key] for key in row_order_keys],
        automargin=True,
        title_standoff=12,
    )
    fig.update_yaxes(
        automargin=True,
        title_standoff=12,
        showspikes=True,
        spikemode="across",
        spikethickness=1,
        spikecolor="#999999",
        spikesnap="cursor",
    )
    fig.update_layout(hovermode="closest")

    return fig


def render_boxplot_fig(fig: "go.Figure", key: str) -> None:
    """
    Render the chart and handle outlier selection.
    """
    st.plotly_chart(
        fig,
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key=key,
        config={"responsive": True},
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
