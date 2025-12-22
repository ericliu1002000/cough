from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from sqlalchemy import text

from settings import get_engine
from utils import load_table_metadata, get_id_column
from exports.subject_profile import to_csv_sections_bytes, to_excel_bytes


st.set_page_config(page_title="受试者档案", layout="wide")
st.title("🧬 受试者全表档案")


def _get_query_param(name: str) -> Optional[str]:
    try:
        params = st.query_params
        if isinstance(params, dict):
            raw = params.get(name)
            if isinstance(raw, list):
                return raw[0] if raw else None
            if raw is not None:
                return str(raw)
    except Exception:
        pass

    try:
        params = st.experimental_get_query_params()
        raw = params.get(name)
        if isinstance(raw, list):
            return raw[0] if raw else None
        if raw is not None:
            return str(raw)
    except Exception:
        return None

    return None


def _quote_ident(name: str) -> str:
    parts = [p.strip("`") for p in str(name).split(".") if p]
    return ".".join(f"`{p.replace('`', '``')}`" for p in parts)




def query_subject_tables(subject_id: Any) -> Dict[str, pd.DataFrame]:
    """
    针对单个受试者，从所有带有 ID 列的表中拉取数据。

    返回:
        {table_name: df_for_subject, ...} 只包含有记录的表。
    """
    results: Dict[str, pd.DataFrame] = {}

    if subject_id is None or subject_id == "":
        return results

    meta = load_table_metadata()
    engine = get_engine()

    for table_name, _cols in meta.items():
        id_col = get_id_column(table_name, meta)
        if not id_col:
            continue

        # 使用 SQLAlchemy 的 text + 命名参数，避免直接把 :sid 拼到原始 SQL 里导致语法错误
        sql = text(
            f"SELECT * FROM {_quote_ident(table_name)} WHERE {_quote_ident(id_col)} = :sid"
        )
        try:
            with engine.connect() as conn:
                df = pd.read_sql(sql, conn, params={"sid": subject_id})
        except Exception as e:
            st.warning(f"读取表 `{table_name}` 失败：{e}")
            continue

        if not df.empty:
            results[table_name] = df

    return results


def main() -> None:
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
    subject_tables = query_subject_tables(subject_id)

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
    export_cols = st.columns(2)
    with export_cols[0]:
        excel_bytes = to_excel_bytes(subject_tables)
        st.download_button(
            "⬇️ 下载 Excel（多表）",
            data=excel_bytes,
            file_name=f"subject_{subject_id}_tables.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with export_cols[1]:
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
