"""Pivot utility helpers shared by views and exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd

from analysis.plugins.methods import AGG_METHODS


@dataclass(frozen=True)
class NestedPivotData:
    row_key_cols: List[str]
    col_key_cols: List[str]
    row_keys: List[Dict[str, str]]
    col_keys: List[Dict[str, str]]
    row_key_tuples: List[Tuple[str, ...]]
    col_key_tuples: List[Tuple[str, ...]]
    value_cols: List[str]
    agg_names: List[str]
    values: Dict[Tuple[Tuple[str, ...], Tuple[str, ...], str, str], Any]


def format_key_label(key: Dict[str, str], sep: str = ", ") -> str:
    """Format a key dict into a label like k1=v1, k2=v2."""
    if not key:
        return ""
    return sep.join(f"{k}={v}" for k, v in key.items())


def _order_row_keys(
    row_keys_df: pd.DataFrame,
    row_key_cols: List[str],
    row_orders: Dict[str, List[str]] | None,
) -> pd.DataFrame:
    """Apply custom ordering to row key values when provided."""
    if not row_key_cols or not row_orders or row_keys_df.empty:
        return row_keys_df

    ordered = row_keys_df.copy()
    order_cols = []
    for col in row_key_cols:
        order_list = row_orders.get(col)
        if not order_list:
            continue
        order_map = {val: idx for idx, val in enumerate(order_list)}
        ordered_col = ordered[col].map(
            lambda v: order_map.get(v, len(order_map))
        )
        order_col = f"_order_{col}"
        ordered[order_col] = ordered_col
        order_cols.append(order_col)

    if not order_cols:
        return row_keys_df

    ordered = ordered.sort_values(order_cols, kind="stable")
    return ordered.drop(columns=order_cols)


def _order_col_keys(
    col_keys_df: pd.DataFrame,
    col_key_cols: List[str],
    col_orders: Dict[str, List[str]] | None,
) -> pd.DataFrame:
    """Apply custom ordering to column key values when provided."""
    if not col_key_cols or not col_orders or col_keys_df.empty:
        return col_keys_df

    ordered = col_keys_df.copy()
    order_cols = []
    for col in col_key_cols:
        order_list = col_orders.get(col)
        if not order_list:
            continue
        order_map = {val: idx for idx, val in enumerate(order_list)}
        ordered_col = ordered[col].map(
            lambda v: order_map.get(v, len(order_map))
        )
        order_col = f"_order_{col}"
        ordered[order_col] = ordered_col
        order_cols.append(order_col)

    if not order_cols:
        return col_keys_df

    ordered = ordered.sort_values(order_cols, kind="stable")
    return ordered.drop(columns=order_cols)


def build_nested_pivot_data(
    df: pd.DataFrame,
    row_key_cols: List[str],
    col_key_cols: List[str],
    value_cols: List[str],
    agg_names: List[str],
    row_orders: Dict[str, List[str]] | None = None,
    col_orders: Dict[str, List[str]] | None = None,
) -> NestedPivotData:
    """Build a nested pivot data structure from a flat DataFrame."""
    row_key_cols = list(row_key_cols or [])
    col_key_cols = list(col_key_cols or [])
    value_cols = list(value_cols or [])
    agg_names = list(agg_names or [])
    if row_orders:
        row_orders = {
            key: [str(v) for v in vals]
            for key, vals in row_orders.items()
            if isinstance(vals, (list, tuple, set))
        }
    if col_orders:
        col_orders = {
            key: [str(v) for v in vals]
            for key, vals in col_orders.items()
            if isinstance(vals, (list, tuple, set))
        }

    work_df = df.copy()
    key_cols = row_key_cols + col_key_cols
    if key_cols:
        work_df = work_df.dropna(subset=key_cols)
        for key_col in key_cols:
            work_df[key_col] = work_df[key_col].astype(str)

    if row_key_cols:
        row_keys_df = work_df[row_key_cols].drop_duplicates()
        row_keys_df = _order_row_keys(row_keys_df, row_key_cols, row_orders)
        row_keys = row_keys_df.to_dict(orient="records")
        row_key_tuples = [
            tuple(rec.get(col, "") for col in row_key_cols) for rec in row_keys
        ]
    else:
        row_keys = [{}]
        row_key_tuples = [()]

    if col_key_cols:
        col_keys_df = work_df[col_key_cols].drop_duplicates()
        col_keys_df = _order_col_keys(col_keys_df, col_key_cols, col_orders)
        col_keys = col_keys_df.to_dict(orient="records")
        col_key_tuples = [
            tuple(rec.get(col, "") for col in col_key_cols) for rec in col_keys
        ]
    else:
        col_keys = [{}]
        col_key_tuples = [()]

    values: Dict[Tuple[Tuple[str, ...], Tuple[str, ...], str, str], Any] = {}
    if value_cols and agg_names:
        group_cols = row_key_cols + col_key_cols
        if group_cols:
            grouped = work_df.groupby(group_cols, dropna=False, sort=False)
        else:
            grouped = [((), work_df)]

        for key, group in grouped:
            key_tuple = key if isinstance(key, tuple) else (key,)
            row_tuple = tuple(key_tuple[: len(row_key_cols)])
            col_tuple = tuple(key_tuple[len(row_key_cols) :])

            for value_col in value_cols:
                series = group[value_col]
                for agg_name in agg_names:
                    agg_func = AGG_METHODS.get(agg_name, "mean")
                    try:
                        if callable(agg_func):
                            agg_val = agg_func(series)
                        else:
                            agg_val = series.agg(agg_func)
                    except Exception:
                        agg_val = None
                    values[(row_tuple, col_tuple, value_col, agg_name)] = agg_val

    return NestedPivotData(
        row_key_cols=row_key_cols,
        col_key_cols=col_key_cols,
        row_keys=row_keys,
        col_keys=col_keys,
        row_key_tuples=row_key_tuples,
        col_key_tuples=col_key_tuples,
        value_cols=value_cols,
        agg_names=agg_names,
        values=values,
    )
