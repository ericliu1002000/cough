"""SQL construction helpers for dynamic queries."""

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from analysis.settings.config import get_business_engine
from analysis.settings.logging import log_exception
from analysis.repositories.metadata_repo import get_id_column


@st.cache_data(ttl=600)
def get_unique_values(table: str, column: str, limit: int = 100) -> List[str]:
    """
    Fetch distinct values for a column to assist UI selection.
    """
    try:
        engine = get_business_engine()
        query = f"SELECT DISTINCT `{column}` FROM `{table}` LIMIT {limit}"
        df = pd.read_sql(query, engine)
        values = df.iloc[:, 0].dropna().astype(str).tolist()
        return sorted(values)
    except Exception:
        log_exception(
            "sql_builder.get_unique_values failed",
            {"table": table, "column": column},
        )
        return []


def format_value_for_sql(val: Any, operator: str) -> str:
    """
    Format a value for SQL based on the operator.
    """
    if operator in ["IS NULL", "IS NOT NULL"]:
        return ""

    def is_number(s: Any) -> bool:
        """Return True when the input can be parsed as a float."""
        try:
            float(str(s))
            return True
        except ValueError:
            return False

    if operator in ["IN", "NOT IN"]:
        if isinstance(val, list):
            items: List[str] = []
            for v in val:
                if is_number(v):
                    items.append(str(v))
                else:
                    items.append(f"'{v}'")
            if not items:
                return "('')"
            return f"({', '.join(items)})"
        return str(val)

    if is_number(val):
        return str(val)
    return f"'{val}'"


def build_sql(
    selected_tables: List[str],
    table_columns_map: Dict[str, List[str]],
    filters: Dict[str, Any],
    subject_blocklist: str,
    meta_data: Dict[str, List[str]],
    group_by: List[Dict[str, Any]] | None = None,
    aggregations: List[Dict[str, Any]] | None = None,
) -> str | None:
    """
    Build a SQL query for the selected tables and filters.
    """
    if not selected_tables:
        return None

    group_by = group_by or []
    aggregations = aggregations or []
    use_group_mode = len(group_by) > 0

    base_table = selected_tables[0]
    base_id_col = get_id_column(base_table, meta_data)
    if not base_id_col:
        st.error(f"❌ 主表 `{base_table}` 中找不到 ID 列")
        return None

    select_clauses: List[str] = []

    if not use_group_mode:
        select_clauses.append(f"`{base_table}`.`{base_id_col}` AS `SUBJECTID`")

        for table in selected_tables:
            cols = table_columns_map.get(table, [])
            for col in cols:
                select_clauses.append(f"`{table}`.`{col}` AS `{table}_{col}`")
    else:
        for item in group_by:
            tbl = item.get("table")
            col = item.get("col")
            if not tbl or not col:
                continue
            alias = item.get("alias") or f"{tbl}_{col}"
            select_clauses.append(f"`{tbl}`.`{col}` AS `{alias}`")

        for agg in aggregations:
            tbl = agg.get("table")
            col = agg.get("col")
            func = (agg.get("func") or "").upper()
            if not tbl or not col or not func:
                continue
            alias = agg.get("alias") or f"{func}_{tbl}_{col}"
            if not func.replace("_", "").isalnum():
                continue
            select_clauses.append(f"{func}(`{tbl}`.`{col}`) AS `{alias}`")

        if not select_clauses:
            st.error("已启用 Group By，但未配置任何分组字段或聚合字段，无法生成 SQL。")
            return None

    select_sql = "SELECT\n    " + ",\n    ".join(select_clauses)

    from_sql = f"\nFROM `{base_table}`"
    join_sql = ""
    for i in range(1, len(selected_tables)):
        current_table = selected_tables[i]
        current_id_col = get_id_column(current_table, meta_data) or "SUBJECTID"
        join_sql += (
            f"\nLEFT JOIN `{current_table}` "
            f"ON `{base_table}`.`{base_id_col}` = `{current_table}`.`{current_id_col}`"
        )

    where_conditions: List[str] = []

    if subject_blocklist:
        ids = [
            x.strip()
            for x in subject_blocklist.replace("，", ",").split("\n")
            if x.strip()
        ]
        if ids:
            id_list_str = "', '".join(ids)
            where_conditions.append(
                f"`{base_table}`.`{base_id_col}` NOT IN ('{id_list_str}')"
            )

    if "conditions" in filters:
        for cond in filters["conditions"]:
            tbl = cond["table"]
            col = cond["col"]
            op = cond["op"]
            val = cond["val"]

            sql_val = format_value_for_sql(val, op)
            clause = f"`{tbl}`.`{col}` {op} {sql_val}"
            where_conditions.append(clause)

    where_sql = ""
    if where_conditions:
        where_sql = "\nWHERE\n  " + "\n  AND ".join(where_conditions)

    group_by_sql = ""
    if use_group_mode and group_by:
        gb_parts: List[str] = []
        for item in group_by:
            tbl = item.get("table")
            col = item.get("col")
            if not tbl or not col:
                continue
            gb_parts.append(f"`{tbl}`.`{col}`")
        if gb_parts:
            group_by_sql = "\nGROUP BY\n  " + ",\n  ".join(gb_parts)

    limit_sql = "\nLIMIT 1000"

    final_sql = f"{select_sql}{from_sql}{join_sql}{where_sql}{group_by_sql}{limit_sql};"
    return final_sql
