"""Streamlit subject profile page."""

from typing import Optional

import pandas as pd
import streamlit as st

from analysis.auth.session import require_login
from analysis.settings.config import TABLE_DESCRIBE_COLUMN
from analysis.settings.logging import log_access
from analysis.exports.subject_profile import to_excel_sections_bytes
from analysis.views.components.page_utils import hide_login_sidebar_entry
from db.services.metadata import get_table_column_display_map
from db.services.subject_profile import (
    fetch_subject_id_candidates,
    query_subject_tables,
    query_table_value_stats,
)


st.set_page_config(page_title="受试者档案", layout="wide")
hide_login_sidebar_entry()
st.title("🧬 受试者全表档案")
st.markdown(
    """
    <style>
    div[data-testid="stDataFrame"] th,
    div[data-testid="stDataFrame"] th div {
        white-space: normal;
        line-height: 1.2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SUBJECT_ID_SUGGESTION_PLACEHOLDER = "<选择受试者 ID>"


def _get_query_param(name: str) -> Optional[str]:
    """Read a query parameter from Streamlit's query params."""
    try:
        params = st.query_params
        if hasattr(params, "get"):
            raw = params.get(name)
        else:
            raw = params[name] if name in params else None
        if isinstance(raw, list):
            return raw[0] if raw else None
        if raw is not None:
            return str(raw)
    except Exception:
        return None

    return None


def _parse_table_describe_columns(raw: str) -> list[str]:
    """Parse a comma-separated list of columns used for table descriptors."""
    if not raw:
        return []
    return [col.strip() for col in raw.split(",") if col.strip()]


def _resolve_table_descriptor(
    df: pd.DataFrame, describe_columns: list[str]
) -> Optional[str]:
    """Return the first non-empty descriptor from configured columns."""
    if df.empty or not describe_columns:
        return None

    col_map = {str(col).lower(): col for col in df.columns}
    for col in describe_columns:
        actual_col = col_map.get(str(col).lower())
        if not actual_col:
            continue
        series = df[actual_col].dropna()
        if series.empty:
            continue
        values = [str(v).strip() for v in series.tolist()]
        values = [v for v in values if v]
        if not values:
            continue
        uniq_values = []
        seen = set()
        for v in values:
            if v in seen:
                continue
            seen.add(v)
            uniq_values.append(v)
        if not uniq_values:
            continue
        if len(uniq_values) == 1:
            return uniq_values[0]
        max_show = 3
        display = ", ".join(uniq_values[:max_show])
        if len(uniq_values) > max_show:
            display = f"{display} 等{len(uniq_values)}项"
        return display
    return None


def _format_empty_value(value: object) -> str:
    if value is None:
        return "空"
    try:
        if pd.isna(value):
            return "空"
    except Exception:
        pass
    if isinstance(value, str) and not value.strip():
        return "空"
    return str(value)


def _get_dataframe_selection(event: object, key: str) -> tuple[int | None, str | None]:
    selection = getattr(event, "selection", None)
    if selection is None:
        state = st.session_state.get(key)
        if isinstance(state, dict):
            selection = state.get("selection")
        else:
            selection = getattr(state, "selection", None)

    if selection is None:
        return None, None

    if hasattr(selection, "rows"):
        rows = selection.rows
        cols = selection.columns
    else:
        rows = selection.get("rows", [])
        cols = selection.get("columns", [])

    row_idx = rows[0] if rows else None
    col_name = cols[0] if cols else None
    return row_idx, col_name


def _get_value_stats(
    table_name: str, col_name: str
) -> tuple[list[dict[str, object]], str | None]:
    cache = st.session_state.setdefault("subject_profile_value_stats", {})
    cache_key = f"{table_name}::{col_name}"
    if cache_key in cache:
        entry = cache[cache_key]
        return entry.get("stats", []), entry.get("error")

    with st.spinner("正在统计列分布..."):
        stats, error = query_table_value_stats(table_name, col_name)
    cache[cache_key] = {"stats": stats, "error": error}
    return stats, error


@st.cache_data(ttl=300)
def _fetch_subject_id_options(limit: int = 20000) -> list[str]:
    return fetch_subject_id_candidates(query="", limit=limit)


def _build_display_column_maps(
    columns: list[str],
    display_name_map: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    label_map: dict[str, str] = {}
    reverse_map: dict[str, str] = {}
    for col in columns:
        display_name = display_name_map.get(col)
        label = f"{col} ({display_name})" if display_name else col
        label_map[col] = label
        reverse_map[label] = col
    return label_map, reverse_map


def main() -> None:
    """Render the subject profile page."""
    require_login()
    log_access("subject_profile")
    display_name_map = get_table_column_display_map(include_hidden=False)
    # 1. 确定当前受试者 ID
    query_subject_id = _get_query_param("subject_id")
    if query_subject_id:
        st.session_state["selected_subject_id"] = query_subject_id
        st.session_state["subject_id_select"] = query_subject_id

    with st.sidebar:
        st.header("受试者选择")
        subject_options = _fetch_subject_id_options(limit=20000)
        current_subject = st.session_state.get("selected_subject_id")
        if current_subject and current_subject not in subject_options:
            subject_options = [str(current_subject)] + subject_options

        select_options = [SUBJECT_ID_SUGGESTION_PLACEHOLDER] + subject_options
        if current_subject and current_subject in select_options:
            st.session_state["subject_id_select"] = current_subject
        elif "subject_id_select" not in st.session_state:
            st.session_state["subject_id_select"] = SUBJECT_ID_SUGGESTION_PLACEHOLDER

        selected_value = st.selectbox(
            "受试者 ID",
            options=select_options,
            key="subject_id_select",
            help="在下拉框里输入可快速筛选",
        )
        if selected_value == SUBJECT_ID_SUGGESTION_PLACEHOLDER:
            st.session_state.pop("selected_subject_id", None)
        else:
            st.session_state["selected_subject_id"] = selected_value

    subject_id = st.session_state.get("selected_subject_id")
    if not subject_id:
        st.info("请在左侧选择受试者 ID，或从分析仪表盘点击散点后跳转到本页面。")
        return

    st.markdown(f"### 当前受试者：`{subject_id}`")

    # 2. 查询所有表
    subject_tables, warnings, skipped_tables = query_subject_tables(subject_id)
    for warn in warnings:
        st.warning(warn)

    if not subject_tables:
        st.warning("在当前配置的表中未找到该受试者的任何记录。")
        if skipped_tables:
            st.markdown("#### ⚠️ 未显示表与原因")
            st.dataframe(
                pd.DataFrame(skipped_tables),
                width="stretch",
                hide_index=True,
            )
        return

    total_rows = sum(len(df) for df in subject_tables.values())
    total_columns = sum(len(df.columns) for df in subject_tables.values())

    c1, c2, c3 = st.columns(3)
    c1.metric("表数量", f"{len(subject_tables)}")
    c2.metric("总行数", f"{total_rows}")
    c3.metric("总列数", f"{total_columns}")

    export_cols = st.columns([3, 1], vertical_alignment="center")
    with export_cols[0]:
        st.markdown("#### 📥 导出数据")
    with export_cols[1]:
        excel_sections_bytes = to_excel_sections_bytes(
            subject_tables, subject_id=str(subject_id)
        )
        st.download_button(
            "⬇️ 下载 Excel（分表）",
            data=excel_sections_bytes,
            file_name=f"subject_{subject_id}_tables_sections.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.markdown(f"共找到 **{len(subject_tables)}** 个表包含该受试者的数据。")
    if skipped_tables:
        with st.expander("查看未显示表与原因"):
            st.dataframe(
                pd.DataFrame(skipped_tables),
                width="stretch",
                hide_index=True,
            )

    st.markdown("#### 📄 表内详情")
    selected_tables = list(subject_tables.keys())
    describe_columns = _parse_table_describe_columns(TABLE_DESCRIBE_COLUMN)

    # 3. 逐表展示
    for table_name in selected_tables:
        df = subject_tables[table_name]
        st.markdown("---")
        descriptor = _resolve_table_descriptor(df, describe_columns)
        if descriptor:
            st.subheader(
                f"表：`{table_name}` [{descriptor}]  （行数：{len(df)}）"
            )
        else:
            st.subheader(f"表：`{table_name}`  （行数：{len(df)}）")

        show_full = False
        if len(df) > 10:
            st.caption(f"默认展示前 10 行，共 {len(df)} 行。")
            show_full = st.checkbox(
                f"显示 `{table_name}` 的全部 {len(df)} 行",
                key=f"show_full_{table_name}",
            )
        display_df = df if show_full or len(df) <= 10 else df.head(10)

        table_display_map = display_name_map.get(table_name, {})
        label_map, reverse_label_map = _build_display_column_maps(
            list(display_df.columns),
            table_display_map,
        )
        display_df = display_df.rename(columns=label_map)

        data_key = f"table_{table_name}_{'full' if show_full else 'head'}"
        event = st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-column",
            key=data_key,
        )

        row_idx, col_name = _get_dataframe_selection(event, data_key)
        if isinstance(col_name, int) and col_name < len(display_df.columns):
            col_name = display_df.columns[col_name]
        display_col = col_name if isinstance(col_name, str) else None
        actual_col = reverse_label_map.get(display_col, display_col)
        if actual_col and actual_col in df.columns:
            st.caption(
                f"当前选择：`{table_name}` / `{display_col}`"
            )

            stats, error = _get_value_stats(table_name, actual_col)
            if error:
                st.warning(f"{error}")
            elif not stats:
                st.info("该列暂无可统计的数据。")
            else:
                show_all_key = f"value_stats_show_all::{table_name}::{actual_col}"
                show_all = st.session_state.get(show_all_key, False)
                display_stats = stats
                if len(stats) > 50 and not show_all:
                    display_stats = stats[:50]
                    st.caption("仅展示前 50 个值。")
                    if st.button(
                        "加载更多",
                        key=f"load_more_{table_name}_{actual_col}",
                    ):
                        show_all = True
                        st.session_state[show_all_key] = True
                        display_stats = stats

                options = []
                for item in display_stats:
                    val_label = _format_empty_value(item.get("value"))
                    record_count = int(item.get("record_count") or 0)
                    subject_count = int(item.get("subject_count") or 0)
                    options.append(
                        f"{val_label}（{record_count} records，{subject_count} patients）"
                    )

                st.selectbox(
                    "value_list",
                    options=options,
                    key=f"value_list_{table_name}_{actual_col}_{int(show_all)}",
                )
        


if __name__ == "__main__":
    main()
