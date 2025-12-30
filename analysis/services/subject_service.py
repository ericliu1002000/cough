"""Subject profile data access helpers."""

from typing import Any, Dict, List, Tuple

import pandas as pd
from sqlalchemy import text

from analysis.settings.config import get_business_engine
from analysis.settings.logging import log_exception
from analysis.repositories.metadata_repo import get_id_column, load_table_metadata


def _quote_ident(name: str) -> str:
    """Return a safely quoted SQL identifier with backticks."""
    parts = [p.strip("`") for p in str(name).split(".") if p]
    return ".".join(f"`{p.replace('`', '``')}`" for p in parts)


def query_subject_tables(subject_id: Any) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Load all rows for a subject across tables that include an ID column.

    Returns:
        (results, warnings)
    """
    results: Dict[str, pd.DataFrame] = {}
    warnings: List[str] = []

    if subject_id is None or subject_id == "":
        return results, warnings

    meta = load_table_metadata(include_hidden=False)
    engine = get_business_engine()

    for table_name, _cols in meta.items():
        id_col = get_id_column(table_name, meta)
        if not id_col:
            continue
        columns = meta.get(table_name, [])
        if not columns:
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
        except Exception as e:
            warnings.append(f"读取表 `{table_name}` 失败：{e}")
            log_exception(
                "subject_service.query_subject_tables failed",
                {"table": table_name, "subject_id": subject_id},
            )
            continue

        if not df.empty:
            results[table_name] = df

    return results, warnings
