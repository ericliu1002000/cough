"""Nested pivot table rendering helpers."""

from __future__ import annotations

import html
from typing import List

import pandas as pd
import streamlit as st

from analysis.views.pivot_utils import (
    NestedPivotData,
    add_p_values_to_pivot,
    build_nested_pivot_data,
    format_key_label,
)


def _format_value(val: object) -> str:
    """Format pivot values for display with fallback for missing data."""
    if val is None:
        return "-"
    try:
        if pd.isna(val):
            return "-"
    except Exception:
        pass
    if isinstance(val, (int,)):
        return str(val)
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)


def _build_row_span_map(
    row_keys: List[dict], row_cols: List[str]
) -> List[dict[int, int]]:
    """Return per-level rowspan maps for contiguous row key prefixes."""
    if not row_cols or not row_keys:
        return []

    span_map: List[dict[int, int]] = []
    for level in range(len(row_cols)):
        level_spans: dict[int, int] = {}
        start = 0
        prev_prefix = None
        for idx in range(len(row_keys) + 1):
            if idx < len(row_keys):
                prefix = tuple(
                    row_keys[idx].get(col, "") for col in row_cols[: level + 1]
                )
            else:
                prefix = None
            if idx == 0:
                prev_prefix = prefix
            if idx == len(row_keys) or prefix != prev_prefix:
                level_spans[start] = idx - start
                start = idx
                prev_prefix = prefix
        span_map.append(level_spans)
    return span_map


def _build_nested_html(data: NestedPivotData, agg_axis: str) -> str:
    """Return HTML for a nested pivot table view."""
    agg_axis = "col" if agg_axis == "col" else "row"
    row_cols = data.row_key_cols
    value_cols = list(data.value_cols or [])
    agg_names = list(data.agg_names or [])
    if not value_cols:
        value_cols = ["值"]
    if not agg_names:
        agg_names = ["-"]

    col_groups = []
    for col_key, col_tuple in zip(data.col_keys, data.col_key_tuples):
        key_label = format_key_label(col_key) or "总体"
        col_groups.append({"col_tuple": col_tuple, "label": key_label})
    if not col_groups:
        col_groups = [{"col_tuple": (), "label": "总体"}]

    col_defs = []
    if agg_axis == "row":
        for group in col_groups:
            for value_col in value_cols:
                if len(value_cols) > 1:
                    label_html = (
                        f"{html.escape(group['label'])}<br>"
                        f"{html.escape(value_col)}"
                    )
                else:
                    label_html = html.escape(group["label"])
                col_defs.append(
                    {
                        "col_tuple": group["col_tuple"],
                        "value_col": value_col,
                        "label_html": label_html,
                    }
                )
    else:
        for group in col_groups:
            for value_col in value_cols:
                for agg_name in agg_names:
                    if len(agg_names) > 1:
                        if len(value_cols) > 1:
                            label_html = (
                                f"{html.escape(agg_name)} | "
                                f"{html.escape(value_col)}"
                            )
                        else:
                            label_html = html.escape(agg_name)
                    elif len(value_cols) > 1:
                        label_html = html.escape(value_col)
                    else:
                        label_html = html.escape(agg_name)
                    col_defs.append(
                        {
                            "col_tuple": group["col_tuple"],
                            "value_col": value_col,
                            "agg_name": agg_name,
                            "label_html": label_html,
                        }
                    )

    header_rows = []
    if agg_axis == "row":
        header_cells = "".join(
            f"<th class='pivot-row-header'>{html.escape(col)}</th>"
            for col in row_cols
        )
        header_cells += "<th class='pivot-agg-header'>统计量</th>"
        header_cells += "".join(
            f"<th class='pivot-col-header'>{col_def['label_html']}</th>"
            for col_def in col_defs
        )
        header_rows.append(f"<tr>{header_cells}</tr>")
    else:
        top_cells = "".join(
            (
                "<th class='pivot-row-header' rowspan='2'>"
                f"{html.escape(col)}</th>"
            )
            for col in row_cols
        )
        group_span = max(len(value_cols), 1) * max(len(agg_names), 1)
        for group in col_groups:
            top_cells += (
                "<th class='pivot-col-header' "
                f"colspan='{group_span}'>"
                f"{html.escape(group['label'])}</th>"
            )
        header_rows.append(f"<tr>{top_cells}</tr>")
        sub_cells = "".join(
            f"<th class='pivot-col-header'>{col_def['label_html']}</th>"
            for col_def in col_defs
        )
        header_rows.append(f"<tr>{sub_cells}</tr>")
    header_html = "".join(header_rows)

    body_rows = []
    row_span_map = _build_row_span_map(data.row_keys, row_cols)
    if agg_axis == "row":
        group_size = max(len(agg_names), 1)
        for row_idx, (row_key, row_tuple) in enumerate(
            zip(data.row_keys, data.row_key_tuples)
        ):
            for agg_idx, agg_name in enumerate(agg_names):
                row_cells = ""
                if agg_idx == 0:
                    for level, col in enumerate(row_cols):
                        level_spans = (
                            row_span_map[level]
                            if level < len(row_span_map)
                            else {}
                        )
                        span = level_spans.get(row_idx)
                        if span:
                            row_cells += (
                                "<th class='pivot-row-header' "
                                f"rowspan='{span * group_size}'>"
                                f"{html.escape(str(row_key.get(col, '')))}</th>"
                            )
                row_cells += (
                    f"<td class='pivot-agg-name'>{html.escape(agg_name)}</td>"
                )
                if agg_name.startswith("P value (ANOVA"):
                    span = max(len(col_groups), 1)
                    for value_col in value_cols:
                        val = None
                        for group in col_groups:
                            val = data.values.get(
                                (
                                    row_tuple,
                                    group["col_tuple"],
                                    value_col,
                                    agg_name,
                                )
                            )
                            if val is not None:
                                break
                        value_text = html.escape(_format_value(val))
                        row_cells += (
                            "<td class='pivot-cell' "
                            f"colspan='{span}'>"
                            f"{value_text}</td>"
                        )
                else:
                    for col_def in col_defs:
                        val = data.values.get(
                            (
                                row_tuple,
                                col_def["col_tuple"],
                                col_def["value_col"],
                                agg_name,
                            )
                        )
                        value_text = html.escape(_format_value(val))
                        row_cells += f"<td class='pivot-cell'>{value_text}</td>"
                row_class = (
                    "pivot-group-start" if agg_idx == 0 else "pivot-group-row"
                )
                body_rows.append(f"<tr class='{row_class}'>{row_cells}</tr>")
    else:
        for row_idx, (row_key, row_tuple) in enumerate(
            zip(data.row_keys, data.row_key_tuples)
        ):
            row_cells = ""
            for level, col in enumerate(row_cols):
                level_spans = (
                    row_span_map[level]
                    if level < len(row_span_map)
                    else {}
                )
                span = level_spans.get(row_idx)
                if span:
                    row_cells += (
                        "<th class='pivot-row-header' "
                        f"rowspan='{span}'>"
                        f"{html.escape(str(row_key.get(col, '')))}</th>"
                    )
            for col_def in col_defs:
                val = data.values.get(
                    (
                        row_tuple,
                        col_def["col_tuple"],
                        col_def["value_col"],
                        col_def["agg_name"],
                    )
                )
                value_text = html.escape(_format_value(val))
                row_cells += f"<td class='pivot-cell'>{value_text}</td>"
            row_class = "pivot-group-start" if row_idx == 0 else "pivot-group-row"
            body_rows.append(f"<tr class='{row_class}'>{row_cells}</tr>")

    table_html = (
        "<div class='pivot-nested-wrapper'>"
        "<table class='pivot-nested-table'>"
        f"<thead>{header_html}</thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )
    return table_html


def render_pivot_nested(
    df: pd.DataFrame,
    index_cols: List[str],
    column_cols: List[str],
    value_cols: List[str],
    agg_names: List[str],
    row_orders: dict[str, list[str]] | None = None,
    col_orders: dict[str, list[str]] | None = None,
    agg_axis: str = "row",
    include_p_values: bool = False,
    p_value_label: str = "P value (ANOVA)",
    control_groups: dict[str, str] | None = None,
    control_label: str = "P value (vs Control)",
) -> NestedPivotData:
    """Render a nested pivot table in Streamlit and return its data model."""
    data = build_nested_pivot_data(
        df,
        row_key_cols=index_cols,
        col_key_cols=column_cols,
        value_cols=value_cols,
        agg_names=agg_names,
        row_orders=row_orders,
        col_orders=col_orders,
    )
    if include_p_values:
        data = add_p_values_to_pivot(
            data,
            df,
            label=p_value_label,
            control_groups=control_groups,
            control_label=control_label,
        )

    if not data.row_keys or not data.col_keys:
        st.info("暂无可展示的数据。")
        return data

    style = """
    <style>
    .pivot-nested-wrapper {
        width: 100%;
        overflow: auto;
        border: 1px solid #ddd;
        border-radius: 6px;
    }
    .pivot-nested-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
    }
    .pivot-nested-table th,
    .pivot-nested-table td {
        border: 1px solid #d0d0d0;
        padding: 6px 8px;
        vertical-align: top;
    }
    .pivot-nested-table thead th {
        background: #f6f6f6;
        text-align: center;
        font-weight: 600;
        white-space: nowrap;
    }
    .pivot-row-header {
        background: #fbfbfb;
        text-align: left;
        white-space: nowrap;
        vertical-align: middle;
    }
    .pivot-col-header {
        min-width: 120px;
    }
    .pivot-cell {
        min-width: 120px;
        text-align: right;
        font-variant-numeric: tabular-nums;
    }
    .pivot-group-start th,
    .pivot-group-start td {
        border-top: 2px solid #c0c0c0;
    }
    .pivot-agg-header {
        background: #f6f6f6;
        text-align: center;
        white-space: nowrap;
    }
    .pivot-agg-name {
        background: #fafafa;
        color: #444;
        white-space: nowrap;
        font-weight: 600;
    }
    </style>
    """

    html_block = style + _build_nested_html(data, agg_axis=agg_axis)
    st.markdown(html_block, unsafe_allow_html=True)
    return data
