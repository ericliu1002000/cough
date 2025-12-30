"""业务元数据配置页面。"""

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
    fetch_business_columns,
    fetch_business_objects,
    sync_business_metadata,
    update_business_columns,
    update_business_objects,
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


def _format_timestamp(value: object) -> str:
    if value is None:
        return "-"
    try:
        parsed = pd.to_datetime(value, errors="coerce")
    except Exception:
        return str(value)
    if pd.isna(parsed):
        return "-"
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _format_timestamp_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    formatted = parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
    return formatted.fillna("-")


def _prepare_editor_df(
    cache_key: str,
    data: list[dict[str, object]],
    order_col: str,
    name_col: str,
    source_col: str | None = None,
) -> pd.DataFrame:
    if cache_key in st.session_state:
        cached = st.session_state[cache_key]
        df = cached.copy() if isinstance(cached, pd.DataFrame) else pd.DataFrame(cached)
    else:
        df = pd.DataFrame(data)

    if df.empty:
        return df

    df[order_col] = _to_int_series(df[order_col])
    df["is_visible"] = _to_bool_series(df["is_visible"])
    if source_col and source_col in df.columns:
        df[source_col] = pd.to_numeric(df[source_col], errors="coerce")
        df = df.sort_values(
            by=[order_col, source_col, name_col],
            ascending=[False, True, True],
            ignore_index=True,
        )
    else:
        df = df.sort_values(
            by=[order_col, name_col],
            ascending=[False, True],
            ignore_index=True,
        )
    return df


def _clear_columns_cache() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("columns_editor_"):
            st.session_state.pop(key, None)


st.set_page_config(page_title="业务元数据配置", layout="wide")
st.title("🧭 业务元数据配置")

try:
    config = get_business_db_config()
except ValueError as exc:
    st.error(str(exc))
    st.stop()

project = config["code"]
system_db = get_system_db_name()

st.markdown(f"用于配置业务表/视图与列的显示名、排序与可见性。系统库: `{system_db}` | 业务库: `{config['database']}`。CURRENT_BUSINESS_CODE: `{project}`")

if st.session_state.pop("metadata_dirty", False):
    _clear_columns_cache()

rule_col, sync_col = st.columns([4, 1], vertical_alignment="center")
with rule_col:
    st.caption("排序规则：`order_index` 越大越靠前，默认 100。")
with sync_col:
    if st.button("立即同步", type="primary", width="stretch"):
        try:
            with st.spinner("正在同步..."):
                result = sync_business_metadata(project)
            st.success(
                "同步完成："
                f"表/视图 {result['objects_scanned']}，"
                f"列 {result['columns_scanned']}"
            )
            _clear_columns_cache()
            st.rerun()
        except Exception as exc:
            st.error(f"同步失败: {exc}")

objects = fetch_business_objects(project, include_hidden=True)
if not objects:
    st.info("暂无元数据，请先点击“立即同步”。")
    st.stop()

objects_df = pd.DataFrame(objects)
if not objects_df.empty:
    objects_df["order_index"] = _to_int_series(objects_df["order_index"])
    objects_df["is_visible"] = _to_bool_series(objects_df["is_visible"])

objects_df = objects_df.sort_values(
    by=["order_index", "object_name"],
    ascending=[False, True],
    ignore_index=True,
)

left_col, right_col = st.columns([1, 4], gap="large")

with left_col:
    st.subheader("表名")
    keyword = st.text_input("搜索表名", placeholder="输入表名或显示名")
    filtered_df = objects_df
    if keyword:
        key_lower = keyword.strip().lower()
        filtered_df = objects_df[
            objects_df["object_name"].str.lower().str.contains(key_lower)
            | objects_df["display_name"]
            .fillna("")
            .str.lower()
            .str.contains(key_lower)
        ]

    if filtered_df.empty:
        st.info("无匹配表。")
        st.stop()

    options = filtered_df["object_name"].tolist()
    option_display = {
        row["object_name"]: f"{row['object_name']} ({row.get('display_name') or '-'})"
        for row in filtered_df.to_dict(orient="records")
    }

    default_object = options[0]
    if "selected_object" in st.session_state:
        selected = st.session_state["selected_object"]
        if selected in options:
            default_object = selected

    selected_object = st.radio(
        "选择表",
        options,
        index=options.index(default_object),
        format_func=lambda name: option_display.get(name, name),
        label_visibility="collapsed",
        key="selected_object",
    )

with right_col:
    st.subheader("表 / 视图信息")
    selected_row = objects_df.loc[
        objects_df["object_name"] == selected_object
    ].iloc[0]

    with st.form("object_form"):
        row1 = st.columns([2, 1, 2], gap="small")
        with row1[0]:
            st.text_input(
                "表/视图名",
                value=str(selected_row["object_name"]),
                disabled=True,
            )
        with row1[1]:
            st.text_input(
                "类型",
                value=str(selected_row["object_type"]),
                disabled=True,
            )
        with row1[2]:
            st.text_input(
                "最近同步",
                value=_format_timestamp(selected_row.get("last_seen_at")),
                disabled=True,
            )

        row2 = st.columns([2, 1, 1, 1], gap="small")
        with row2[0]:
            display_name = st.text_input(
                "显示名", value=selected_row.get("display_name") or ""
            )
        with row2[1]:
            order_index = st.number_input(
                "排序权重",
                value=int(selected_row.get("order_index") or DEFAULT_ORDER_INDEX),
                step=1,
            )
        with row2[2]:
            is_visible = st.checkbox(
                "显示", value=bool(selected_row.get("is_visible"))
            )
        with row2[3]:
            save_object = st.form_submit_button(
                "保存表信息", width="stretch"
            )

    if save_object:
        updated = update_business_objects(
            project,
            [
                {
                    "id": selected_row["id"],
                    "display_name": display_name,
                    "order_index": order_index,
                    "is_visible": is_visible,
                }
            ],
        )
        st.success(f"已更新 {updated} 条表信息。")
        st.rerun()

    st.subheader("列配置")
    columns = fetch_business_columns(
        project, selected_object, include_hidden=True
    )
    if not columns:
        st.info("该表暂无列信息，请先同步后再配置。")
    else:
        editor_key = f"columns_editor_{project}_{selected_object}"
        cache_key = f"{editor_key}_data"
        columns_df = _prepare_editor_df(
            cache_key, columns, "order_index", "column_name", "source_order_index"
        )

        display_columns = columns_df.copy()
        display_columns["last_seen_at"] = _format_timestamp_series(
            display_columns["last_seen_at"]
        )
        display_columns = display_columns[
            [
                "column_name",
                "display_name",
                "order_index",
                "is_visible",
                "last_seen_at",
            ]
        ]

        edited_columns = st.data_editor(
            display_columns,
            hide_index=True,
            width="stretch",
            num_rows="fixed",
            column_config={
                "column_name": st.column_config.TextColumn("列名", disabled=True),
                "display_name": st.column_config.TextColumn("显示名"),
                "order_index": st.column_config.NumberColumn("排序权重"),
                "is_visible": st.column_config.CheckboxColumn("显示"),
                "last_seen_at": st.column_config.TextColumn("最近同步", disabled=True),
            },
            key=editor_key,
        )
        edits = edited_columns.set_index("column_name")
        updated_full = columns_df.copy()
        updated_full["display_name"] = (
            updated_full["column_name"]
            .map(edits["display_name"])
            .combine_first(updated_full["display_name"])
        )
        updated_full["order_index"] = (
            updated_full["column_name"]
            .map(edits["order_index"])
            .combine_first(updated_full["order_index"])
        )
        updated_full["is_visible"] = (
            updated_full["column_name"]
            .map(edits["is_visible"])
            .combine_first(updated_full["is_visible"])
        )
        st.session_state[cache_key] = updated_full

        if st.button("保存列配置"):
            payload = updated_full[
                ["id", "display_name", "order_index", "is_visible"]
            ].to_dict(orient="records")
            updated = update_business_columns(
                project, selected_object, payload
            )
            st.success(f"已更新 {updated} 条列配置。")
            st.session_state.pop(editor_key, None)
            st.rerun()
