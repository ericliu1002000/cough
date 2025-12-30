"""列快速配置页面。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from db.services.db_config import get_business_db_config, get_system_db_name
from db.services.metadata import (
    DEFAULT_ORDER_INDEX,
    fetch_column_name_counts,
    fetch_columns_by_name,
    update_business_columns,
)

load_dotenv(BASE_DIR / ".env")


def _to_int_series(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .fillna(DEFAULT_ORDER_INDEX)
        .astype(int)
    )


def _to_bool_series(series: pd.Series) -> pd.Series:
    return series.fillna(0).astype(bool)


def _format_timestamp_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    formatted = parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
    return formatted.fillna("-")


st.set_page_config(page_title="列快速配置", layout="wide")
st.title("🔎 列快速配置")

try:
    config = get_business_db_config()
except ValueError as exc:
    st.error(str(exc))
    st.stop()

project = config["code"]
system_db = get_system_db_name()

st.markdown(
    f"系统库: `{system_db}` | 业务库: `{config['database']}`"
)
st.caption(f"CURRENT_BUSINESS_CODE: `{project}`")

st.markdown("选择列名后，将展示该列所在的所有表，可快速调整权重与显示开关。")

saved_message = st.session_state.pop("metadata_column_saved", None)
if saved_message:
    st.success(saved_message)

column_rows = fetch_column_name_counts(project)
if not column_rows:
    st.info("暂无列元数据，请先上传业务数据并同步。")
    st.stop()

column_names = [row["column_name"] for row in column_rows]
count_map = {row["column_name"]: row["cnt"] for row in column_rows}

selected_column = st.selectbox(
    "输入列名搜索",
    options=column_names,
    format_func=lambda name: f"{name} ({count_map.get(name, 0)})",
)

columns = fetch_columns_by_name(project, selected_column)
if not columns:
    st.info("未找到该列的配置记录。")
    st.stop()

columns_df = pd.DataFrame(columns).set_index("id")
columns_df["order_index"] = _to_int_series(columns_df["order_index"])
columns_df["is_visible"] = _to_bool_series(columns_df["is_visible"])
columns_df = columns_df.sort_values(
    by=["order_index", "object_name"],
    ascending=[False, True],
    ignore_index=False,
)

display_df = columns_df[
    ["object_name", "display_name", "order_index", "is_visible", "last_seen_at"]
].copy()
display_df["last_seen_at"] = _format_timestamp_series(
    display_df["last_seen_at"]
)

editor_key = f"column_fast_editor_{project}_{selected_column}"
bulk_visible_key = f"{editor_key}_bulk_visible"
bulk_weight_key = f"{editor_key}_bulk_weight"
bulk_weight_input_key = f"{editor_key}_bulk_weight_input"

bulk_cols = st.columns([1, 1, 2, 1], gap="small")
with bulk_cols[0]:
    if st.button("全选显示", key=f"{editor_key}_all_visible"):
        st.session_state[bulk_visible_key] = True
        st.session_state.pop(editor_key, None)
        st.rerun()
with bulk_cols[1]:
    if st.button("全不选", key=f"{editor_key}_none_visible"):
        st.session_state[bulk_visible_key] = False
        st.session_state.pop(editor_key, None)
        st.rerun()
with bulk_cols[2]:
    bulk_weight = st.number_input(
        "统一权重",
        value=int(
            st.session_state.get(bulk_weight_key, DEFAULT_ORDER_INDEX)
        ),
        step=1,
        key=bulk_weight_input_key,
    )
with bulk_cols[3]:
    if st.button("应用权重", key=f"{editor_key}_apply_weight"):
        st.session_state[bulk_weight_key] = int(bulk_weight)
        st.session_state.pop(editor_key, None)
        st.rerun()

if bulk_visible_key in st.session_state:
    display_df["is_visible"] = bool(st.session_state[bulk_visible_key])
if bulk_weight_key in st.session_state:
    display_df["order_index"] = int(st.session_state[bulk_weight_key])
display_df = display_df.sort_values(
    by=["order_index", "object_name"],
    ascending=[False, True],
    ignore_index=False,
)

edited = st.data_editor(
    display_df,
    hide_index=True,
    width="stretch",
    num_rows="fixed",
    column_config={
        "object_name": st.column_config.TextColumn("表名", disabled=True),
        "display_name": st.column_config.TextColumn("显示名", disabled=True),
        "order_index": st.column_config.NumberColumn("排序权重"),
        "is_visible": st.column_config.CheckboxColumn("显示"),
        "last_seen_at": st.column_config.TextColumn("最近同步", disabled=True),
    },
    key=editor_key,
)

if st.button("保存列配置", type="primary"):
    updated = 0
    edited_reset = edited.reset_index()
    if "id" not in edited_reset.columns:
        edited_reset.insert(0, "id", columns_df.index)
    for object_name, group in edited_reset.groupby("object_name"):
        rows = group[
            ["id", "display_name", "order_index", "is_visible"]
        ].to_dict(orient="records")
        updated += update_business_columns(project, object_name, rows)
    st.session_state["metadata_column_saved"] = (
        f"已更新 {updated} 条列配置。"
        if updated > 0
        else "未更新任何列，请检查选择。"
    )
    st.session_state["metadata_dirty"] = True
    st.session_state.pop(editor_key, None)
    st.session_state.pop(bulk_visible_key, None)
    st.session_state.pop(bulk_weight_key, None)
    st.rerun()
