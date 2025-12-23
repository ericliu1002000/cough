from __future__ import annotations

from typing import Dict, List

import pandas as pd
import streamlit as st

from analysis_methods import AGG_METHODS


def _apply_row_order(
    pivot: pd.DataFrame,
    index_cols: List[str],
    row_orders: Dict[str, List[str]] | None,
) -> pd.DataFrame:
    if not row_orders or pivot.empty or not index_cols:
        return pivot

    if isinstance(pivot.index, pd.MultiIndex):
        level_name_to_idx = {
            name: idx for idx, name in enumerate(pivot.index.names)
        }
        row_level_indices = [
            level_name_to_idx[name]
            for name in index_cols
            if name in level_name_to_idx
        ]
        if not row_level_indices:
            return pivot

        idx_frame = pivot.index.to_frame(index=False)
        idx_frame.columns = [
            f"level_{i}" for i in range(idx_frame.shape[1])
        ]

        order_cols_for = {}
        for field in index_cols:
            order_list = row_orders.get(field)
            if not order_list:
                continue
            level_idx = level_name_to_idx.get(field)
            if level_idx is None:
                continue
            order_map = {str(val): idx for idx, val in enumerate(order_list)}
            order_cols_for[f"level_{level_idx}"] = order_map

        if not order_cols_for:
            return pivot

        order_cols = []
        for level_col, order_map in order_cols_for.items():
            idx_frame[level_col] = idx_frame[level_col].astype(str)
            order_name = f"_order_{level_col}"
            idx_frame[order_name] = idx_frame[level_col].map(
                lambda v: order_map.get(str(v), len(order_map))
            )
            order_cols.append(order_name)

        new_order = idx_frame.sort_values(order_cols, kind="stable").index
        return pivot.iloc[new_order]

    if len(index_cols) == 1 and pivot.index.name in row_orders:
        order_map = {
            str(val): idx
            for idx, val in enumerate(row_orders[pivot.index.name])
        }
        order_key = [
            order_map.get(str(v), len(order_map)) for v in pivot.index
        ]
        pos_key = list(range(len(pivot.index)))
        sort_df = pd.DataFrame(
            {"order": order_key, "pos": pos_key}, index=pivot.index
        )
        new_index = sort_df.sort_values(["order", "pos"]).index
        return pivot.loc[new_index]

    return pivot


def _apply_col_order(
    pivot: pd.DataFrame,
    column_cols: List[str],
    col_orders: dict[str, list[str]] | None,
) -> pd.DataFrame:
    if not col_orders or pivot.empty or not column_cols:
        return pivot

    if isinstance(pivot.columns, pd.MultiIndex):
        level_name_to_idx = {
            name: idx for idx, name in enumerate(pivot.columns.names)
        }
        col_level_indices = [
            level_name_to_idx[name]
            for name in column_cols
            if name in level_name_to_idx
        ]
        if not col_level_indices:
            return pivot

        col_frame = pivot.columns.to_frame(index=False)
        col_frame.columns = [
            f"level_{i}" for i in range(col_frame.shape[1])
        ]
        col_level_cols = [f"level_{i}" for i in col_level_indices]
        group_cols = [c for c in col_frame.columns if c not in col_level_cols]

        order_cols_for = {}
        for field in column_cols:
            order_list = col_orders.get(field)
            if not order_list:
                continue
            level_idx = level_name_to_idx.get(field)
            if level_idx is None:
                continue
            order_map = {str(val): idx for idx, val in enumerate(order_list)}
            order_cols_for[f"level_{level_idx}"] = order_map

        if not order_cols_for:
            return pivot

        new_order: list[int] = []
        if group_cols:
            grouped = col_frame.groupby(group_cols, sort=False, dropna=False)
            for _, idxs in grouped.groups.items():
                sub = col_frame.loc[idxs].copy()
                order_cols = []
                for level_col, order_map in order_cols_for.items():
                    sub_col = sub[level_col].map(
                        lambda v: order_map.get(str(v), len(order_map))
                    )
                    order_name = f"_order_{level_col}"
                    sub[order_name] = sub_col
                    order_cols.append(order_name)
                if order_cols:
                    sub = sub.sort_values(order_cols, kind="stable")
                new_order.extend(sub.index.tolist())
        else:
            sub = col_frame.copy()
            order_cols = []
            for level_col, order_map in order_cols_for.items():
                sub_col = sub[level_col].map(
                    lambda v: order_map.get(str(v), len(order_map))
                )
                order_name = f"_order_{level_col}"
                sub[order_name] = sub_col
                order_cols.append(order_name)
            if order_cols:
                sub = sub.sort_values(order_cols, kind="stable")
            new_order = sub.index.tolist()

        return pivot.iloc[:, new_order]

    if len(column_cols) == 1 and pivot.columns.name in col_orders:
        order_map = {
            str(val): idx
            for idx, val in enumerate(col_orders[pivot.columns.name])
        }
        order_key = [
            order_map.get(str(v), len(order_map)) for v in pivot.columns
        ]
        pos_key = list(range(len(pivot.columns)))
        sort_df = pd.DataFrame(
            {"order": order_key, "pos": pos_key}, index=pivot.columns
        )
        new_cols = sort_df.sort_values(["order", "pos"]).index
        return pivot.loc[:, new_cols]

    return pivot


def build_pivot_table(
    df: pd.DataFrame,
    index_cols: List[str],
    column_cols: List[str],
    value_cols: List[str],
    agg_names: List[str],
    row_orders: Dict[str, List[str]] | None = None,
    col_orders: dict[str, list[str]] | None = None,
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

    pivot = _apply_row_order(pivot, index_cols, row_orders)
    pivot = _apply_col_order(pivot, column_cols, col_orders)
    return pivot


def render_pivot_classic(
    df: pd.DataFrame,
    index_cols: List[str],
    column_cols: List[str],
    value_cols: List[str],
    agg_names: List[str],
    row_orders: Dict[str, List[str]] | None = None,
    col_orders: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    pivot = build_pivot_table(
        df,
        index_cols=index_cols,
        column_cols=column_cols,
        value_cols=value_cols,
        agg_names=agg_names,
        row_orders=row_orders,
        col_orders=col_orders,
    )
    st.dataframe(pivot, width="stretch")
    return pivot
