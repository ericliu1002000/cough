from __future__ import annotations

import html
from typing import List

import pandas as pd
import streamlit as st

from views.pivot_utils import NestedPivotData, build_nested_pivot_data, format_key_label


def _format_value(val: object) -> str:
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


def _build_nested_html(data: NestedPivotData) -> str:
    row_cols = data.row_key_cols
    col_cols = data.col_key_cols
    value_cols = data.value_cols
    agg_names = data.agg_names

    col_defs = []
    for col_key, col_tuple in zip(data.col_keys, data.col_key_tuples):
        key_label = format_key_label(col_key)
        if not key_label:
            key_label = "总体"
        for value_col in value_cols:
            if len(value_cols) > 1:
                label_html = (
                    f"{html.escape(key_label)}<br>"
                    f"{html.escape(value_col)}"
                )
            else:
                label_html = html.escape(key_label)
            col_defs.append(
                {
                    "col_tuple": col_tuple,
                    "value_col": value_col,
                    "label_html": label_html,
                }
            )

    header_cells = "".join(
        f"<th class='pivot-row-header'>{html.escape(col)}</th>"
        for col in row_cols
    )
    header_cells += "".join(
        f"<th class='pivot-col-header'>{col_def['label_html']}</th>"
        for col_def in col_defs
    )
    header_html = f"<tr>{header_cells}</tr>"

    body_rows = []
    for row_key, row_tuple in zip(data.row_keys, data.row_key_tuples):
        row_cells = "".join(
            f"<th class='pivot-row-header'>{html.escape(str(row_key[col]))}</th>"
            for col in row_cols
        )
        for col_def in col_defs:
            mini_rows = []
            for agg_name in agg_names:
                val = data.values.get(
                    (row_tuple, col_def["col_tuple"], col_def["value_col"], agg_name)
                )
                value_text = html.escape(_format_value(val))
                mini_rows.append(
                    "<tr>"
                    f"<td class='pivot-agg-name'>{html.escape(agg_name)}</td>"
                    f"<td class='pivot-agg-val'>{value_text}</td>"
                    "</tr>"
                )
            mini_html = (
                "<table class='pivot-mini-table'>"
                + "".join(mini_rows)
                + "</table>"
            )
            row_cells += f"<td class='pivot-cell'>{mini_html}</td>"
        body_rows.append(f"<tr>{row_cells}</tr>")

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
    row_order: List[str] | None = None,
) -> NestedPivotData:
    data = build_nested_pivot_data(
        df,
        row_key_cols=index_cols,
        col_key_cols=column_cols,
        value_cols=value_cols,
        agg_names=agg_names,
        row_order=row_order,
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
    }
    .pivot-col-header {
        min-width: 140px;
    }
    .pivot-cell {
        min-width: 140px;
    }
    .pivot-mini-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 11px;
    }
    .pivot-mini-table td {
        padding: 2px 4px;
        border-bottom: 1px solid #e2e2e2;
    }
    .pivot-mini-table tr:last-child td {
        border-bottom: none;
    }
    .pivot-agg-name {
        color: #666;
        white-space: nowrap;
    }
    .pivot-agg-val {
        text-align: right;
        font-variant-numeric: tabular-nums;
    }
    </style>
    """

    html_block = style + _build_nested_html(data)
    st.markdown(html_block, unsafe_allow_html=True)
    return data
