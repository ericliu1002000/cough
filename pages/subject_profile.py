"""Streamlit subject profile page."""

from typing import Any, Optional

import pandas as pd
import streamlit as st

from analysis.auth.session import require_login
from analysis.settings.logging import log_access
from analysis.exports.subject_profile import (
    to_csv_sections_bytes,
    to_excel_bytes,
    to_excel_sections_bytes,
)
from analysis.services.subject_service import query_subject_tables


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
    subject_tables, warnings = query_subject_tables(subject_id)
    for warn in warnings:
        st.warning(warn)

    if not subject_tables:
        st.warning("在当前配置的表中未找到该受试者的任何记录。")
        return

    total_rows = sum(len(df) for df in subject_tables.values())
    summary_rows = [
        {"Table": name, "Rows": len(df), "Columns": len(df.columns)}
        for name, df in subject_tables.items()
    ]
    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["Rows", "Table"], ascending=[False, True]
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("表数量", f"{len(subject_tables)}")
    c2.metric("总行数", f"{total_rows}")
    c3.metric("总列数", f"{summary_df['Columns'].sum()}")

    st.markdown("#### 📦 数据概览")
    st.dataframe(summary_df, width="stretch", hide_index=True)

    st.markdown("#### 📥 导出数据")
    export_cols = st.columns(3)
    with export_cols[0]:
        excel_bytes = to_excel_bytes(subject_tables)
        st.download_button(
            "⬇️ 下载 Excel（多表）",
            data=excel_bytes,
            file_name=f"subject_{subject_id}_tables.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
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
    with export_cols[2]:
        csv_bytes = to_csv_sections_bytes(subject_tables)
        st.download_button(
            "⬇️ 下载 CSV（分表）",
            data=csv_bytes,
            file_name=f"subject_{subject_id}_tables.csv",
            mime="text/csv",
        )

    st.markdown(f"共找到 **{len(subject_tables)}** 个表包含该受试者的数据。")

    st.markdown("#### 📄 表内详情")
    table_filter = st.text_input("按表名筛选", value="")
    table_names = list(subject_tables.keys())
    if table_filter:
        table_names = [
            name for name in table_names if table_filter.lower() in name.lower()
        ]

    selected_tables = st.multiselect(
        "选择要查看的表",
        options=table_names,
        default=table_names,
    )

    # 3. 逐表展示
    for table_name in selected_tables:
        df = subject_tables[table_name]
        st.markdown("---")
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
