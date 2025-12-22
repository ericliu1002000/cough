from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st

from analysis_methods import AGG_METHODS


def _apply_row_order(
    pivot: pd.DataFrame, row_order: List[str] | None
) -> pd.DataFrame:
    if not row_order or pivot.empty:
        return pivot

    order_map = {str(val): idx for idx, val in enumerate(row_order)}
    if isinstance(pivot.index, pd.MultiIndex):
        level0 = pivot.index.get_level_values(0)
    else:
        level0 = pivot.index

    order_key = [order_map.get(str(v), len(order_map)) for v in level0]
    pos_key = list(range(len(pivot.index)))
    sort_df = pd.DataFrame({"order": order_key, "pos": pos_key}, index=pivot.index)
    new_index = sort_df.sort_values(["order", "pos"]).index
    return pivot.loc[new_index]


def build_pivot_table(
    df: pd.DataFrame,
    index_cols: List[str],
    column_cols: List[str],
    value_cols: List[str],
    agg_names: List[str],
    row_order: List[str] | None = None,
) -> pd.DataFrame:
    p_src = df.copy()
    for v in value_cols:
        if v in p_src.columns:
            p_src[v] = pd.to_numeric(p_src[v], errors="coerce")

    aggfunc_map = {
        v: [AGG_METHODS.get(a, "mean") for a in agg_names] for v in value_cols
    }

    pivot = pd.pivot_table(
        p_src,
        index=index_cols,
        columns=column_cols,
        values=value_cols,
        aggfunc=aggfunc_map,
        sort=False,
    )

    return _apply_row_order(pivot, row_order)


def render_pivot_classic(
    df: pd.DataFrame,
    index_cols: List[str],
    column_cols: List[str],
    value_cols: List[str],
    agg_names: List[str],
    row_order: List[str] | None = None,
) -> pd.DataFrame:
    pivot = build_pivot_table(
        df,
        index_cols=index_cols,
        column_cols=column_cols,
        value_cols=value_cols,
        agg_names=agg_names,
        row_order=row_order,
    )
    st.dataframe(pivot, width="stretch")
    return pivot
