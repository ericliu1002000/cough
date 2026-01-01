"""Helpers for reading and writing analysis calculation configs."""

from __future__ import annotations

from typing import Any, Iterable, MutableMapping

GRAPH_VERSION = 2
GRAPH_ROOT_ID = "raw"
OUTPUT_ALL_COLUMNS = "*"


def normalize_calc_config(raw_cfg: Any) -> dict[str, Any]:
    """Normalize calculation config into a dictionary."""
    if raw_cfg is None:
        return {}
    if isinstance(raw_cfg, list):
        return {"calc_rules": raw_cfg}
    if isinstance(raw_cfg, dict):
        return dict(raw_cfg)
    return {}


def _listify(value: Any, default: list[str] | None = None) -> list[Any]:
    """Return a list from value, falling back to default when missing."""
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    if value is None:
        return list(default or [])
    return [value]


def _normalize_dict_of_lists(value: Any) -> dict[str, list[str]]:
    """Normalize dict values into string lists."""
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, list[str]] = {}
    for key, vals in value.items():
        if isinstance(vals, (list, tuple, set)):
            normalized[str(key)] = list(vals)
        else:
            normalized[str(key)] = []
    return normalized


def build_graph_from_legacy(calc_cfg: dict[str, Any]) -> dict[str, Any]:
    """Build a DAG schema from legacy calculation config."""
    nodes: list[dict[str, Any]] = []
    nodes.append(
        {
            "id": GRAPH_ROOT_ID,
            "type": "source",
            "inputs": [],
            "inputs_cols": [],
            "outputs_cols": [OUTPUT_ALL_COLUMNS],
            "params": {},
        }
    )
    baseline_cfg = calc_cfg.get("baseline", {})
    if isinstance(baseline_cfg, dict):
        subj_col = baseline_cfg.get("subj_col")
        visit_col = baseline_cfg.get("visit_col")
        baseline_val = baseline_cfg.get("baseline_val")
        target_cols = _listify(baseline_cfg.get("target_cols"), [])
        for idx, target in enumerate(target_cols, start=1):
            if not target:
                continue
            node_id = f"baseline_{idx}"
            inputs_cols = [c for c in (subj_col, visit_col, target) if c]
            nodes.append(
                {
                    "id": node_id,
                    "type": "baseline",
                    "inputs": [],
                    "inputs_cols": inputs_cols,
                    "outputs_cols": [f"{target}_BL"],
                    "params": {
                        "subj_col": subj_col,
                        "visit_col": visit_col,
                        "baseline_val": baseline_val,
                        "target_col": target,
                    },
                }
            )

    calc_rules = calc_cfg.get("calc_rules", [])
    if not isinstance(calc_rules, list):
        calc_rules = []
    for idx, rule in enumerate(calc_rules, start=1):
        if not isinstance(rule, dict):
            continue
        node_id = f"derive_{idx}"
        inputs_cols = _listify(rule.get("cols"), [])
        outputs_cols = [rule.get("name")] if rule.get("name") else []
        nodes.append(
            {
                "id": node_id,
                "type": "derive",
                "inputs": [],
                "inputs_cols": inputs_cols,
                "outputs_cols": outputs_cols,
                "params": dict(rule),
            }
        )

    exclusions = calc_cfg.get("exclusions", [])
    if not isinstance(exclusions, list):
        exclusions = []
    for idx, rule in enumerate(exclusions, start=1):
        if not isinstance(rule, dict):
            continue
        field = rule.get("field")
        values = _listify(rule.get("values"), [])
        node_id = f"filter_{idx}"
        nodes.append(
            {
                "id": node_id,
                "type": "filter",
                "inputs": [],
                "inputs_cols": [field] if field else [],
                "outputs_cols": [],
                "params": {
                    "field": field,
                    "values": values,
                    "op": rule.get("op", "not_in"),
                },
            }
        )

    agg_rules = (
        calc_cfg.get("aggregations")
        or calc_cfg.get("aggregate_rules")
        or []
    )
    if not isinstance(agg_rules, list):
        agg_rules = []
    for idx, rule in enumerate(agg_rules, start=1):
        if not isinstance(rule, dict):
            continue
        group_by = _listify(rule.get("group_by"), [])
        metrics = rule.get("metrics", [])
        outputs_cols = []
        input_cols = list(group_by)
        if isinstance(metrics, list):
            for metric in metrics:
                if not isinstance(metric, dict):
                    continue
                col = metric.get("col")
                if col:
                    input_cols.append(col)
                name = metric.get("name")
                if name:
                    outputs_cols.append(name)
        node_id = f"aggregate_{idx}"
        nodes.append(
            {
                "id": node_id,
                "type": "aggregate",
                "inputs": [],
                "inputs_cols": input_cols,
                "outputs_cols": outputs_cols,
                "params": dict(rule),
            }
        )

    node_order = {node["id"]: idx for idx, node in enumerate(nodes)}
    output_to_node: dict[str, str] = {}
    for node in nodes:
        for col in node.get("outputs_cols", []):
            if col and col != OUTPUT_ALL_COLUMNS:
                output_to_node[str(col)] = node["id"]

    for node in nodes:
        if node.get("type") == "source":
            continue
        deps: list[str] = []
        for col in node.get("inputs_cols", []):
            if not col:
                continue
            producer = output_to_node.get(str(col))
            if producer and producer != node["id"]:
                deps.append(producer)
        if not deps:
            node["inputs"] = [GRAPH_ROOT_ID]
            continue
        seen: set[str] = set()
        ordered: list[str] = []
        for dep in deps:
            if dep in seen:
                continue
            seen.add(dep)
            ordered.append(dep)
        ordered.sort(key=lambda dep: node_order.get(dep, 0))
        node["inputs"] = ordered

    return {
        "version": GRAPH_VERSION,
        "root": GRAPH_ROOT_ID,
        "leaf": nodes[-1]["id"] if nodes else GRAPH_ROOT_ID,
        "nodes": nodes,
    }


def ensure_calc_graph(calc_cfg: dict[str, Any]) -> dict[str, Any]:
    """Ensure a calculation config has a DAG graph."""
    updated = dict(calc_cfg)
    graph = updated.get("graph")
    if (
        not isinstance(graph, dict)
        or graph.get("version") != GRAPH_VERSION
        or "nodes" not in graph
    ):
        updated["graph"] = build_graph_from_legacy(updated)
    return updated


def _parse_node_index(node_id: str, prefix: str) -> int | None:
    """Return a zero-based index for node ids like prefix_1."""
    if not node_id.startswith(prefix):
        return None
    suffix = node_id[len(prefix) :]
    if not suffix.isdigit():
        return None
    return int(suffix) - 1


def _remove_indices(items: list[Any], indices: set[int]) -> list[Any]:
    """Remove items by index, keeping the rest."""
    if not indices:
        return items
    return [item for idx, item in enumerate(items) if idx not in indices]


def cascade_delete_targets(
    calc_cfg: dict[str, Any],
    targets: Iterable[str],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Remove nodes that produce targets and their dependents."""
    target_list = [str(t) for t in targets if t]
    if not target_list:
        return calc_cfg, {"nodes": [], "outputs": []}

    cfg_with_graph = ensure_calc_graph(calc_cfg)
    graph = cfg_with_graph.get("graph", {})
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, list):
        return calc_cfg, {"nodes": [], "outputs": []}

    nodes_by_id = {
        str(node.get("id")): node for node in nodes if node.get("id") is not None
    }
    output_to_node: dict[str, str] = {}
    for node in nodes:
        node_id = str(node.get("id") or "")
        for col in node.get("outputs_cols", []):
            if col and col != OUTPUT_ALL_COLUMNS:
                output_to_node[str(col)] = node_id

    dependents: dict[str, list[str]] = {node_id: [] for node_id in nodes_by_id}
    for node in nodes_by_id.values():
        node_id = str(node.get("id") or "")
        for dep in _listify(node.get("inputs")):
            dep_id = str(dep)
            if dep_id in dependents:
                dependents[dep_id].append(node_id)

    start_nodes: set[str] = set()
    for target in target_list:
        if target in nodes_by_id:
            start_nodes.add(target)
            continue
        node_id = output_to_node.get(target)
        if node_id:
            start_nodes.add(node_id)

    if not start_nodes:
        return calc_cfg, {"nodes": [], "outputs": []}

    to_visit = list(start_nodes)
    nodes_to_remove: set[str] = set()
    while to_visit:
        node_id = to_visit.pop()
        if node_id in nodes_to_remove:
            continue
        nodes_to_remove.add(node_id)
        for child in dependents.get(node_id, []):
            if child not in nodes_to_remove:
                to_visit.append(child)

    removed_outputs: list[str] = []
    for node_id in nodes_to_remove:
        node = nodes_by_id.get(node_id, {})
        for col in node.get("outputs_cols", []):
            if col and col != OUTPUT_ALL_COLUMNS:
                removed_outputs.append(str(col))

    updated = dict(calc_cfg)
    updated.pop("graph", None)

    baseline_cfg = updated.get("baseline")
    baseline_indices: set[int] = set()
    derive_indices: set[int] = set()
    filter_indices: set[int] = set()
    aggregate_indices: set[int] = set()

    for node_id in nodes_to_remove:
        baseline_idx = _parse_node_index(node_id, "baseline_")
        derive_idx = _parse_node_index(node_id, "derive_")
        filter_idx = _parse_node_index(node_id, "filter_")
        aggregate_idx = _parse_node_index(node_id, "aggregate_")
        if baseline_idx is not None:
            baseline_indices.add(baseline_idx)
        if derive_idx is not None:
            derive_indices.add(derive_idx)
        if filter_idx is not None:
            filter_indices.add(filter_idx)
        if aggregate_idx is not None:
            aggregate_indices.add(aggregate_idx)

    if baseline_indices and isinstance(baseline_cfg, dict):
        target_cols = _listify(baseline_cfg.get("target_cols"), [])
        baseline_cfg = dict(baseline_cfg)
        baseline_cfg["target_cols"] = _remove_indices(
            list(target_cols), baseline_indices
        )
        updated["baseline"] = baseline_cfg

    calc_rules = updated.get("calc_rules", [])
    if isinstance(calc_rules, list) and derive_indices:
        updated["calc_rules"] = _remove_indices(calc_rules, derive_indices)

    exclusions = updated.get("exclusions", [])
    if isinstance(exclusions, list) and filter_indices:
        updated["exclusions"] = _remove_indices(exclusions, filter_indices)

    agg_rules = updated.get("aggregations")
    if isinstance(agg_rules, list) and aggregate_indices:
        updated["aggregations"] = _remove_indices(agg_rules, aggregate_indices)
    elif "aggregate_rules" in updated:
        agg_rules = updated.get("aggregate_rules")
        if isinstance(agg_rules, list) and aggregate_indices:
            updated["aggregate_rules"] = _remove_indices(agg_rules, aggregate_indices)

    updated = ensure_calc_graph(updated)
    removed_outputs = sorted({col for col in removed_outputs if col})
    removed_nodes = sorted(nodes_to_remove)
    return updated, {"nodes": removed_nodes, "outputs": removed_outputs}


def build_calculation_payload(
    state: MutableMapping[str, Any],
    *,
    default_agg: list[str] | None = None,
    default_agg_axis: str = "row",
) -> dict[str, Any]:
    """Build the calculation payload from session state."""
    row_orders_map = state.get("pivot_row_orders", {})
    if not isinstance(row_orders_map, dict):
        row_orders_map = {}
    row_fields = state.get("pivot_index", [])
    row_fields_list = _listify(row_fields, [])
    if row_fields_list:
        row_orders_map = {
            k: list(v) if isinstance(v, (list, tuple, set)) else []
            for k, v in row_orders_map.items()
            if k in row_fields_list
        }
    else:
        row_orders_map = {}

    note = state.get("calc_note_input")
    if note is None:
        note = state.get("calc_note", "")

    agg_axis = state.get("pivot_agg_axis", default_agg_axis)
    if agg_axis not in {"row", "col"}:
        agg_axis = default_agg_axis

    payload = {
        "baseline": state.get("baseline_config", {}),
        "calc_rules": state.get("calc_rules", []),
        "note": note,
        "exclusions": state.get("exclusions", []),
        "pivot": {
            "index": _listify(state.get("pivot_index"), []),
            "columns": _listify(state.get("pivot_columns"), []),
            "values": _listify(state.get("pivot_values"), []),
            "agg": _listify(state.get("pivot_aggs"), default_agg),
            "agg_axis": agg_axis,
            "row_order": row_orders_map,
            "col_order": _normalize_dict_of_lists(state.get("pivot_col_order", {})),
            "uniform_control_group": state.get("uniform_control_group"),
        },
    }
    return ensure_calc_graph(payload)


def apply_calculation_config(
    state: MutableMapping[str, Any],
    raw_cfg: Any,
    *,
    default_agg: list[str] | None = None,
    default_agg_axis: str = "row",
) -> None:
    """Apply a calculation config to session state."""
    calc_cfg = ensure_calc_graph(normalize_calc_config(raw_cfg))
    state["calc_rules"] = calc_cfg.get("calc_rules", [])
    state["calc_note"] = calc_cfg.get("note", "")
    state["exclusions"] = calc_cfg.get("exclusions", [])
    state["pivot_config"] = calc_cfg.get("pivot", {})
    state["baseline_config"] = calc_cfg.get("baseline", {})
    state["calc_graph"] = calc_cfg.get("graph", {})

    p_cfg = state["pivot_config"]
    if not isinstance(p_cfg, dict):
        p_cfg = {}
        state["pivot_config"] = p_cfg

    raw_agg = p_cfg.get("agg", default_agg or [])
    state["pivot_aggs"] = _listify(raw_agg, default_agg)
    state["pivot_index"] = _listify(p_cfg.get("index"), [])
    state["pivot_columns"] = _listify(p_cfg.get("columns"), [])
    state["pivot_values"] = _listify(p_cfg.get("values"), [])

    agg_axis_cfg = p_cfg.get("agg_axis", default_agg_axis)
    if agg_axis_cfg not in {"row", "col"}:
        agg_axis_cfg = default_agg_axis
    state["pivot_agg_axis"] = agg_axis_cfg

    row_order_cfg = p_cfg.get("row_order", {})
    row_orders: dict[str, list[str]] = {}
    if isinstance(row_order_cfg, dict):
        if "field" in row_order_cfg and "values" in row_order_cfg:
            field = row_order_cfg.get("field")
            values = row_order_cfg.get("values", [])
            if field:
                row_orders[str(field)] = _listify(values, [])
        else:
            row_orders = _normalize_dict_of_lists(row_order_cfg)
    state["pivot_row_orders"] = row_orders
    state.pop("pivot_row_order_field", None)
    state.pop("pivot_row_order_values", None)
    state.pop("pivot_agg_axis_ui", None)

    state["pivot_col_order"] = _normalize_dict_of_lists(p_cfg.get("col_order", {}))

    control_group_cfg = p_cfg.get("uniform_control_group")
    if isinstance(control_group_cfg, dict):
        state["uniform_control_group"] = control_group_cfg
    else:
        state.pop("uniform_control_group", None)
