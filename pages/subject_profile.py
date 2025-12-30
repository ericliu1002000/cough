"""Streamlit subject profile page."""

from typing import Any, Optional

import pandas as pd
import streamlit as st

from analysis.auth.session import require_login
from analysis.settings.config import TABLE_DESCRIBE_COLUMN
from analysis.settings.logging import log_access
from analysis.exports.subject_profile import to_excel_sections_bytes
from db.services.subject_profile import query_subject_tables


st.set_page_config(page_title="受试者档案", layout="wide")
st.title("🧬 受试者全表档案")


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


def main() -> None:
    """Render the subject profile page."""
    require_login()
    log_access("subject_profile")
    # 1. 确定当前受试者 ID
    query_subject_id = _get_query_param("subject_id")
    if query_subject_id:
        st.session_state["selected_subject_id"] = query_subject_id

    subject_id = st.session_state.get("selected_subject_id")

    with st.sidebar:
        st.header("受试者选择")
        subject_id = st.text_input(
            "受试者 ID",
            value=str(subject_id) if subject_id is not None else "",
            help="可从分析仪表盘点击散点后跳转，也可以在此手动输入。",
        )
        if st.button("加载受试者档案"):
            st.session_state["selected_subject_id"] = subject_id

    if not subject_id:
        st.info("请在左侧输入受试者 ID，或从分析仪表盘点击散点后跳转到本页面。")
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
    table_order_map = {
        name: idx for idx, name in enumerate(subject_tables.keys())
    }

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

        if len(df) <= 10:
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.caption(f"默认展示前 10 行，共 {len(df)} 行。")
            show_full = st.checkbox(
                f"显示 `{table_name}` 的全部 {len(df)} 行",
                key=f"show_full_{table_name}",
            )
            if show_full:
                st.dataframe(df, width="stretch", hide_index=True)
            else:
                st.dataframe(df.head(10), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
