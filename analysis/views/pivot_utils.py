"""Pivot utility helpers shared by views and exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np

from analysis.plugins.methods import AGG_METHODS
from analysis.plugins.methods.composite import (
    calculate_anova_f_test,
    calculate_t_test_from_summary,
)


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


def add_p_values_to_pivot(
    data: NestedPivotData,
    df: pd.DataFrame,
    *,
    label: str = "P value (ANOVA)",
    control_groups: dict[str, str] | None = None,
    control_label: str = "P value (vs Control)",
) -> NestedPivotData:
    """Return pivot data augmented with ANOVA and control-group P values."""
    if not data.col_key_cols or not data.value_cols:
        return data

    work_df = df.copy()
    key_cols = list(data.row_key_cols or []) + list(data.col_key_cols or [])
    for key_col in key_cols:
        if key_col in work_df.columns:
            work_df[key_col] = work_df[key_col].astype(str)

    anova_values: dict[tuple[tuple[str, ...], str, str], float | None] = {}
    control_values: dict[
        tuple[tuple[str, ...], str, str, str], float | None
    ] = {}

    control_groups = control_groups or {}
    control_groups = {
        str(k): str(v)
        for k, v in control_groups.items()
        if v is not None and str(v) != ""
    }

    if data.row_key_cols:
        grouped = work_df.groupby(data.row_key_cols, dropna=False, sort=False)
    else:
        grouped = [((), work_df)]

    for row_key, group in grouped:
        if data.row_key_cols:
            row_tuple = row_key if isinstance(row_key, tuple) else (row_key,)
        else:
            row_tuple = ()
        for col_dim in data.col_key_cols:
            if col_dim not in group.columns:
                continue
            for value_col in data.value_cols:
                if value_col not in group.columns:
                    continue
                subset = group[[col_dim, value_col]].copy()
                subset = subset.dropna(subset=[col_dim, value_col])
                subset[col_dim] = subset[col_dim].astype(str)
                subset[value_col] = pd.to_numeric(
                    subset[value_col], errors="coerce"
                )
                _, p_val = calculate_anova_f_test(subset, col_dim, value_col)
                anova_values[(row_tuple, value_col, col_dim)] = p_val

                control_val = control_groups.get(col_dim)
                if not control_val:
                    continue
                ctrl_series = subset.loc[
                    subset[col_dim] == control_val, value_col
                ].dropna()
                if ctrl_series.empty:
                    continue
                ctrl_mean = float(ctrl_series.mean())
                ctrl_sd = float(ctrl_series.std(ddof=1))
                ctrl_n = int(ctrl_series.count())
                if ctrl_n <= 1 or np.isnan(ctrl_sd):
                    continue

                for group_val in subset[col_dim].dropna().unique().tolist():
                    group_val = str(group_val)
                    if group_val == control_val:
                        control_values[
                            (row_tuple, value_col, col_dim, group_val)
                        ] = np.nan
                        continue
                    grp_series = subset.loc[
                        subset[col_dim] == group_val, value_col
                    ].dropna()
                    grp_mean = float(grp_series.mean()) if not grp_series.empty else np.nan
                    grp_sd = float(grp_series.std(ddof=1)) if not grp_series.empty else np.nan
                    grp_n = int(grp_series.count())
                    if grp_n <= 1 or np.isnan(grp_sd):
                        p_ctrl = np.nan
                    else:
                        _, p_ctrl = calculate_t_test_from_summary(
                            mean_trt=grp_mean,
                            mean_placebo=ctrl_mean,
                            sd_trt=grp_sd,
                            sd_placebo=ctrl_sd,
                            n_trt=grp_n,
                            n_placebo=ctrl_n,
                        )
                    control_values[
                        (row_tuple, value_col, col_dim, group_val)
                    ] = p_ctrl

    if not anova_values and not control_values:
        return data

    col_dim_count = len(data.col_key_cols)
    anova_label_map = {
        col_dim: label
        if col_dim_count == 1
        else f"{label} ({col_dim})"
        for col_dim in data.col_key_cols
    }
    control_label_map: dict[str, str] = {}
    if control_groups:
        for col_dim in data.col_key_cols:
            control_val = control_groups.get(col_dim)
            if not control_val:
                continue
            if col_dim_count == 1:
                control_label_map[col_dim] = f"{control_label}: {control_val}"
            else:
                control_label_map[col_dim] = (
                    f"{control_label} ({col_dim}: {control_val})"
                )

    extra_agg_names = [
        name
        for name in list(anova_label_map.values())
        + list(control_label_map.values())
        if name not in data.agg_names
    ]
    if not extra_agg_names:
        return data

    col_key_map = {
        col_tuple: col_key
        for col_key, col_tuple in zip(data.col_keys, data.col_key_tuples)
    }
    new_values = dict(data.values)
    for row_tuple in data.row_key_tuples:
        for col_tuple in data.col_key_tuples:
            col_key = col_key_map.get(col_tuple, {})
            for value_col in data.value_cols:
                for col_dim, agg_name in anova_label_map.items():
                    p_val = anova_values.get((row_tuple, value_col, col_dim))
                    if p_val is None:
                        p_val = np.nan
                    new_values[(row_tuple, col_tuple, value_col, agg_name)] = p_val
                for col_dim, agg_name in control_label_map.items():
                    group_val = str(col_key.get(col_dim, ""))
                    p_val = control_values.get(
                        (row_tuple, value_col, col_dim, group_val)
                    )
                    if p_val is None:
                        p_val = np.nan
                    new_values[(row_tuple, col_tuple, value_col, agg_name)] = p_val

    return NestedPivotData(
        row_key_cols=data.row_key_cols,
        col_key_cols=data.col_key_cols,
        row_keys=data.row_keys,
        col_keys=data.col_keys,
        row_key_tuples=data.row_key_tuples,
        col_key_tuples=data.col_key_tuples,
        value_cols=data.value_cols,
        agg_names=list(data.agg_names) + extra_agg_names,
        values=new_values,
    )
