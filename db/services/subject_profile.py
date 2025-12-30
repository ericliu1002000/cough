"""Subject profile data access helpers (DB layer)."""

from typing import Any, Dict, List, Tuple

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


def query_subject_tables(
    subject_id: Any,
) -> Tuple[Dict[str, pd.DataFrame], List[str], List[Dict[str, str]]]:
    """
    Load all rows for a subject across tables that include an ID column.

    Returns:
        (results, warnings)
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
