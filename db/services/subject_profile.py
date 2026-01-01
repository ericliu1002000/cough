"""Subject profile data access helpers (DB layer)."""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import text

from analysis.settings.constants import SUBJECT_ID_ALIASES
from analysis.settings.logging import log_exception
from db.services.db_config import get_business_db_config, get_business_engine
from db.services.metadata import fetch_business_objects, get_table_columns_map


def _quote_ident(name: str) -> str:
    """Return a safely quoted SQL identifier with backticks."""
    parts = [p.strip("`") for p in str(name).split(".") if p]
    return ".".join(f"`{p.replace('`', '``')}`" for p in parts)


def _get_id_column(table_name: str, meta_data: Dict[str, List[str]]) -> str | None:
    """Find the first matching ID column in a table based on known aliases."""
    available_columns = meta_data.get(table_name, [])
    for alias in SUBJECT_ID_ALIASES:
        if alias in available_columns:
            return alias
    return None


def _resolve_table_name(
    meta_data: Dict[str, List[str]],
    table_name: str,
) -> str | None:
    """Resolve a table name using case-insensitive matching."""
    if not table_name:
        return None
    for name in meta_data.keys():
        if str(name).lower() == table_name.lower():
            return str(name)
    return None


def _resolve_subject_id_column(columns: List[str]) -> str | None:
    """Resolve subject ID column using case-insensitive aliases."""
    col_map = {str(col).lower(): str(col) for col in columns}
    for alias in SUBJECT_ID_ALIASES:
        match = col_map.get(str(alias).lower())
        if match:
            return match
    return None


def query_subject_tables(
    subject_id: Any,
) -> Tuple[Dict[str, pd.DataFrame], List[str], List[Dict[str, str]]]:
    """
    Load all rows for a subject across tables that include an ID column.

    Returns:
        (results, warnings, skipped)
    """
    results: Dict[str, pd.DataFrame] = {}
    warnings: List[str] = []
    skipped: List[Dict[str, str]] = []

    if subject_id is None or subject_id == "":
        return results, warnings, skipped

    config = get_business_db_config()
    objects = fetch_business_objects(config["code"], include_hidden=True)
    if not objects:
        warnings.append("未找到业务表元数据，请先同步或上传业务数据。")
        return results, warnings, skipped

    meta = get_table_columns_map(include_hidden=False)
    if not meta:
        warnings.append("未找到表结构元数据，请先同步或上传业务数据。")
        return results, warnings, skipped

    engine = get_business_engine()

    for obj in objects:
        table_name = obj.get("object_name")
        if not table_name:
            continue
        if not obj.get("is_visible", 0):
            skipped.append({"Table": str(table_name), "Reason": "已隐藏"})
            continue

        columns = meta.get(table_name, [])
        if not columns:
            skipped.append({"Table": str(table_name), "Reason": "缺少列元数据"})
            continue

        id_col = _get_id_column(table_name, meta)
        if not id_col:
            skipped.append({"Table": str(table_name), "Reason": "缺少 ID 列"})
            continue
        if id_col not in columns:
            columns = [id_col] + columns

        select_cols = ", ".join(_quote_ident(col) for col in columns)
        sql = text(
            f"SELECT {select_cols} FROM {_quote_ident(table_name)} "
            f"WHERE {_quote_ident(id_col)} = :sid"
        )
        try:
            with engine.connect() as conn:
                df = pd.read_sql(sql, conn, params={"sid": subject_id})
        except Exception as exc:
            warnings.append(f"读取表 `{table_name}` 失败：{exc}")
            skipped.append(
                {"Table": str(table_name), "Reason": f"读取失败: {exc}"}
            )
            log_exception(
                "subject_profile.query_subject_tables failed",
                {"table": table_name, "subject_id": subject_id},
            )
            continue

        if df.empty:
            skipped.append({"Table": str(table_name), "Reason": "该受试者无记录"})
            continue
        results[table_name] = df

    return results, warnings, skipped


def query_table_value_stats(
    table_name: str, value_col: str
) -> Tuple[List[Dict[str, object]], Optional[str]]:
    """
    Return value distribution stats for a column across the whole table.

    Returns:
        (records, error_message)
    """
    if not table_name or not value_col:
        return [], "表名或列名为空"

    meta = get_table_columns_map(include_hidden=False)
    columns = meta.get(table_name, [])
    if not columns:
        return [], "未找到表结构元数据"

    if value_col not in columns:
        return [], "列不在元数据中"

    id_col = _get_id_column(table_name, meta)
    if not id_col:
        return [], "缺少 ID 列"

    sql = text(
        "SELECT "
        f"{_quote_ident(value_col)} AS value, "
        "COUNT(*) AS record_count, "
        f"COUNT(DISTINCT {_quote_ident(id_col)}) AS subject_count "
        f"FROM {_quote_ident(table_name)} "
        f"GROUP BY {_quote_ident(value_col)} "
        "ORDER BY record_count DESC"
    )
    engine = get_business_engine()
    try:
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn)
    except Exception as exc:
        log_exception(
            "subject_profile.query_table_value_stats failed",
            {"table": table_name, "column": value_col},
        )
        return [], f"查询失败: {exc}"

    if df.empty:
        return [], None

    records = df.to_dict(orient="records")
    return records, None


def fetch_subject_id_candidates(
    query: Optional[str] = None,
    limit: int = 50,
    table_name: str = "adsl",
) -> List[str]:
    """Return distinct subject IDs from the specified table."""
    meta = get_table_columns_map(include_hidden=False)
    if not meta:
        return []

    actual_table = _resolve_table_name(meta, table_name)
    if not actual_table:
        return []

    columns = meta.get(actual_table, [])
    id_col = _resolve_subject_id_column(columns)
    if not id_col:
        return []

    table_sql = _quote_ident(actual_table)
    col_sql = _quote_ident(id_col)
    sql = (
        f"SELECT DISTINCT {col_sql} AS subject_id "
        f"FROM {table_sql} "
        f"WHERE {col_sql} IS NOT NULL "
    )
    params: Dict[str, object] = {"limit": int(limit)}
    trimmed = (query or "").strip()
    if trimmed:
        sql += f"AND CAST({col_sql} AS CHAR) LIKE :pattern "
        params["pattern"] = f"%{trimmed}%"
    sql += f"ORDER BY {col_sql} LIMIT :limit"

    engine = get_business_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(sql), params).fetchall()
    except Exception as exc:
        log_exception(
            "subject_profile.fetch_subject_id_candidates failed",
            {"table": actual_table, "column": id_col, "query": trimmed},
        )
        return []

    return [str(row[0]) for row in rows if row and row[0] is not None]
