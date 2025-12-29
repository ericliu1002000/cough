"""Metadata access helpers for table schemas and ID columns."""

from typing import Dict, List

import streamlit as st

from db.services.metadata import get_table_columns_map


def load_table_metadata(include_hidden: bool = False) -> Dict[str, List[str]]:
    """读取表结构信息（来自系统库元数据配置）。"""
    meta = get_table_columns_map(include_hidden=include_hidden)
    if not meta:
        st.error("未找到表结构元数据，请先同步或上传业务数据。")
    return meta


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
