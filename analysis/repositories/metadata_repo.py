"""Metadata access helpers for table schemas and ID columns."""

import json
from pathlib import Path
from typing import Dict, List

import streamlit as st


def load_table_metadata() -> Dict[str, List[str]]:
    """
    Load table schema info from analysis/db/table_columns.json.
    """
    json_path = Path(__file__).resolve().parents[1] / "db" / "table_columns.json"

    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    st.error(
        "未找到表结构文件: "
        f"{json_path}。请先运行 `python -m analysis.db.exp_table_columns`。"
    )
    return {}


def get_id_column(table_name: str, meta_data: Dict[str, List[str]]) -> str | None:
    """
    Find the first matching ID column in a table based on known aliases.
    """
    from analysis.settings.constants import SUBJECT_ID_ALIASES

    available_columns = meta_data.get(table_name, [])
    for alias in SUBJECT_ID_ALIASES:
        if alias in available_columns:
            return alias
    return None
