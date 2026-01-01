"""Execute calculation DAGs against pandas DataFrames."""

from __future__ import annotations

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


def _summarize_outputs(outputs: list[Any], max_cols: int = 3) -> str:
    """Summarize output columns for display."""
    clean = [str(c) for c in outputs if c]
    if not clean:
        return ""
    if len(clean) <= max_cols:
        return ", ".join(clean)
    return ", ".join(clean[:max_cols]) + f" +{len(clean) - max_cols}"


def build_graphviz_dot(graph: dict[str, Any]) -> str:
    """Build a Graphviz DOT string for the DAG."""
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, list):
        return "digraph G {}"

    lines = ["digraph G {", "rankdir=LR;", "node [shape=box];"]
    for node in nodes:
        node_id = str(node.get("id") or "")
        if not node_id:
            continue
        node_type = str(node.get("type") or "")
        outputs = _summarize_outputs(node.get("outputs_cols", []))
        label_parts = [node_id]
        if node_type:
            label_parts.append(node_type)
        if outputs:
            label_parts.append(outputs)
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
    if not field or not values or field not in df.columns:
        return df

    op = str(params.get("op") or "not_in").strip().lower().replace(" ", "_")
    value_set = {str(v) for v in values}
    series = df[field].astype(str)

    if op in {"in"}:
        return df[series.isin(value_set)]
    return df[~series.isin(value_set)]


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
