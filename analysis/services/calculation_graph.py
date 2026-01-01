"""Execute calculation DAGs against pandas DataFrames."""

from __future__ import annotations

import re
from collections import deque
from typing import Any, Iterable

import pandas as pd
import streamlit as st

from analysis.services.analysis_service import apply_baseline_mapping, apply_calculations
from analysis.services.calculation_config import GRAPH_ROOT_ID


def _listify(value: Any) -> list[Any]:
    """Return a list from value, falling back to an empty list."""
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    if value is None:
        return []
    return [value]


def _unique(values: Iterable[str]) -> list[str]:
    """Return a list without duplicates, preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _escape_dot_label(value: str) -> str:
    """Escape label content for Graphviz."""
    return (
        value.replace("\\", "\\\\")
        .replace("\"", "\\\"")
        .replace("\n", "\\n")
    )


def _summarize_items(
    items: list[Any],
    *,
    max_items: int | None = 2,
    empty: str = "-",
) -> str:
    """Summarize items for display."""
    clean = [str(c) for c in items if c]
    if not clean:
        return empty
    if max_items is None or len(clean) <= max_items:
        return ", ".join(clean)
    return ", ".join(clean[:max_items]) + f" +{len(clean) - max_items}"


def _node_display_name(node: dict[str, Any], *, max_items: int | None = 2) -> str:
    """Return the display name for a node."""
    node_type = str(node.get("type") or "")
    if node_type == "source" or node.get("id") == GRAPH_ROOT_ID:
        return "dataset"
    outputs = [
        c for c in _listify(node.get("outputs_cols")) if c and c != "*"
    ]
    if outputs:
        return _summarize_items(outputs, max_items=max_items)
    if node_type == "filter":
        params = node.get("params") or {}
        field = params.get("field")
        if field:
            return f"filter: {field}"
        return "filter"
    return str(node.get("id") or "-")


def _node_dependency_names(node: dict[str, Any]) -> list[Any]:
    """Return dependency variable names for a node."""
    inputs_cols = [c for c in _listify(node.get("inputs_cols")) if c]
    if inputs_cols:
        return inputs_cols
    return _listify(node.get("inputs"))


def _normalize_filter_op(op: Any) -> str:
    """Normalize filter operators."""
    raw = str(op or "not_in").strip().lower()
    raw = raw.replace(" ", "_")
    mapping = {
        "notin": "not_in",
        "not-in": "not_in",
        "not_in": "not_in",
        "in": "in",
        "=": "eq",
        "==": "eq",
        "eq": "eq",
        "!=": "ne",
        "<>": "ne",
        "ne": "ne",
        ">": "gt",
        "gt": "gt",
        ">=": "ge",
        "ge": "ge",
        "gte": "ge",
        "<": "lt",
        "lt": "lt",
        "<=": "le",
        "le": "le",
        "lte": "le",
        "like": "like",
        "contains": "like",
        "is_null": "is_null",
        "null": "is_null",
        "empty": "is_null",
        "isnull": "is_null",
        "为空": "is_null",
        "is_not_null": "is_not_null",
        "not_null": "is_not_null",
        "notnull": "is_not_null",
        "not_empty": "is_not_null",
        "不为空": "is_not_null",
    }
    return mapping.get(raw, raw)


def _sql_like_to_regex(pattern: str) -> str:
    """Convert SQL LIKE pattern to regex."""
    escaped = re.escape(pattern)
    regex = escaped.replace(r"\%", ".*").replace(r"\_", ".")
    return f"^{regex}$"


def build_graphviz_dot(graph: dict[str, Any]) -> str:
    """Build a Graphviz DOT string for the DAG."""
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, list):
        return "digraph G {}"

    lines = [
        "digraph G {",
        "rankdir=LR;",
        "graph [ranksep=0.35, nodesep=0.25];",
        "node [shape=box, fontsize=10, margin=\"0.04,0.02\"];",
    ]
    for node in nodes:
        node_id = str(node.get("id") or "")
        if not node_id:
            continue
        node_type = str(node.get("type") or "")
        node_name = _node_display_name(node, max_items=2)
        depends = _summarize_items(_node_dependency_names(node), max_items=3)
        label_parts = [
            f"node: {node_name}",
            f"type: {node_type or '-'}",
            f"depends: {depends}",
        ]
        label = _escape_dot_label("\n".join(label_parts))
        shape = "oval" if node_type == "source" or node_id == GRAPH_ROOT_ID else "box"
        lines.append(f"\"{_escape_dot_label(node_id)}\" [label=\"{label}\", shape={shape}];")

    for node in nodes:
        node_id = str(node.get("id") or "")
        if not node_id:
            continue
        deps = _listify(node.get("inputs"))
        for dep in deps:
            dep_id = str(dep)
            if dep_id:
                lines.append(
                    f"\"{_escape_dot_label(dep_id)}\" -> \"{_escape_dot_label(node_id)}\";"
                )

    lines.append("}")
    return "\n".join(lines)


def build_dependency_rows(graph: dict[str, Any]) -> list[dict[str, str]]:
    """Build dependency rows for display."""
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, list):
        return []

    rows: list[dict[str, str]] = []
    for node in nodes:
        node_id = str(node.get("id") or "")
        node_type = str(node.get("type") or "")
        inputs = ", ".join(_unique([str(v) for v in _listify(node.get("inputs")) if v]))
        inputs_cols = ", ".join(
            _unique([str(v) for v in _listify(node.get("inputs_cols")) if v])
        )
        outputs_cols = ", ".join(
            _unique([str(v) for v in _listify(node.get("outputs_cols")) if v])
        )
        rows.append(
            {
                "node": node_id,
                "type": node_type,
                "depends_on": inputs,
                "inputs": inputs_cols,
                "outputs": outputs_cols,
            }
        )
    return rows


def _topo_sort(nodes: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Return nodes in topological order, or None when a cycle exists."""
    nodes_by_id = {
        node.get("id"): node
        for node in nodes
        if node.get("id") is not None
    }
    in_degree = {node_id: 0 for node_id in nodes_by_id}
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in nodes_by_id}

    for node in nodes_by_id.values():
        node_id = node.get("id")
        deps = _listify(node.get("inputs"))
        for dep in deps:
            if dep not in nodes_by_id:
                continue
            adjacency[dep].append(node_id)
            in_degree[node_id] += 1

    queue = deque(
        [
            node.get("id")
            for node in nodes
            if node.get("id") in in_degree and in_degree[node.get("id")] == 0
        ]
    )
    ordered: list[dict[str, Any]] = []
    while queue:
        node_id = queue.popleft()
        ordered.append(nodes_by_id[node_id])
        for neighbor in adjacency.get(node_id, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(ordered) != len(nodes_by_id):
        return None
    return ordered


def _apply_filter(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Apply a filter node to the dataframe."""
    field = params.get("field")
    values = _listify(params.get("values"))
    if not field or field not in df.columns:
        return df

    op = _normalize_filter_op(params.get("op"))
    series = df[field]

    if op in {"is_null", "is_not_null"}:
        null_mask = series.isna()
        if series.dtype == object:
            null_mask |= series.astype(str).str.strip().eq("")
        return df[~null_mask] if op == "is_not_null" else df[null_mask]

    if op in {"in", "not_in"}:
        cleaned_values = []
        for val in values:
            try:
                if pd.isna(val):
                    continue
            except Exception:
                if val is None:
                    continue
            cleaned_values.append(val)
        if not cleaned_values:
            return df
        if pd.api.types.is_numeric_dtype(series):
            series_num = pd.to_numeric(series, errors="coerce")
            values_num = (
                pd.to_numeric(pd.Series(cleaned_values), errors="coerce")
                .dropna()
                .tolist()
            )
            if not values_num:
                return df
            mask = series_num.isin(values_num)
        else:
            series_str = series.astype(str)
            value_set = {str(v) for v in cleaned_values}
            mask = series_str.isin(value_set)
        return df[mask] if op == "in" else df[~mask]

    if op in {"gt", "ge", "lt", "le"}:
        if not values:
            return df
        target = pd.to_numeric(values[0], errors="coerce")
        if pd.isna(target):
            return df
        series_num = pd.to_numeric(series, errors="coerce")
        if op == "gt":
            return df[series_num > target]
        if op == "ge":
            return df[series_num >= target]
        if op == "lt":
            return df[series_num < target]
        if op == "le":
            return df[series_num <= target]

    if op == "eq":
        if not values:
            return df
        target = values[0]
        if pd.api.types.is_numeric_dtype(series):
            target_num = pd.to_numeric(target, errors="coerce")
            if pd.isna(target_num):
                return df
            series_num = pd.to_numeric(series, errors="coerce")
            return df[series_num == target_num]
        series_str = series.astype(str)
        return df[series_str == str(target)]

    if op == "ne":
        if not values:
            return df
        target = values[0]
        if pd.api.types.is_numeric_dtype(series):
            target_num = pd.to_numeric(target, errors="coerce")
            if pd.isna(target_num):
                return df
            series_num = pd.to_numeric(series, errors="coerce")
            return df[series_num != target_num]
        series_str = series.astype(str)
        return df[series_str != str(target)]

    if op == "like":
        if not values:
            return df
        pattern = str(values[0])
        if not pattern:
            return df
        regex = _sql_like_to_regex(pattern)
        series_str = series.astype(str)
        return df[series_str.str.contains(regex, regex=True, na=False)]

    return df


def _apply_aggregate(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Apply an aggregate node to the dataframe."""
    group_by = [c for c in _listify(params.get("group_by")) if c in df.columns]
    metrics = params.get("metrics", [])
    if not group_by or not isinstance(metrics, list):
        return df

    aggregations: dict[str, tuple[str, str]] = {}
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        col = metric.get("col")
        func = metric.get("fn") or metric.get("func")
        if not col or not func or col not in df.columns:
            continue
        name = metric.get("name") or f"{func}_{col}"
        aggregations[name] = (col, func)

    if not aggregations:
        return df

    try:
        return df.groupby(group_by).agg(**aggregations).reset_index()
    except Exception as exc:
        st.warning(f"Aggregate node failed: {exc}")
        return df


def run_calculation_graph(df: pd.DataFrame, graph: dict[str, Any]) -> pd.DataFrame:
    """Execute a calculation graph against the dataframe."""
    if not graph or not isinstance(graph, dict):
        return df
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, list) or not nodes:
        return df

    ordered_nodes = _topo_sort(nodes)
    if ordered_nodes is None:
        st.warning("Detected a cycle in the DAG; falling back to node order.")
        ordered_nodes = nodes

    result = df.copy()
    for node in ordered_nodes:
        node_type = node.get("type")
        if node_type == "source" or node.get("id") == GRAPH_ROOT_ID:
            continue
        params = node.get("params") or {}

        if node_type == "baseline":
            target_cols = params.get("target_cols")
            target_col = params.get("target_col")
            if target_cols is None:
                target_cols = [target_col] if target_col else []
            target_cols = [c for c in _listify(target_cols) if c]
            if not target_cols:
                continue
            config = {
                "subj_col": params.get("subj_col"),
                "visit_col": params.get("visit_col"),
                "baseline_val": params.get("baseline_val"),
                "target_cols": target_cols,
            }
            result = apply_baseline_mapping(result, config)
            continue

        if node_type == "derive":
            rule = dict(params)
            if not rule.get("name"):
                outputs = node.get("outputs_cols") or []
                if outputs:
                    rule["name"] = outputs[0]
            if not rule.get("cols"):
                rule["cols"] = node.get("inputs_cols", [])
            if not rule.get("name") or not rule.get("cols") or not rule.get("method"):
                continue
            result = apply_calculations(result, [rule])
            continue

        if node_type == "filter":
            result = _apply_filter(result, params)
            continue

        if node_type == "aggregate":
            result = _apply_aggregate(result, params)

    return result
