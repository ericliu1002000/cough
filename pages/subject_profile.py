import pandas as pd
import streamlit as st
from typing import Any, Dict, List

from sqlalchemy import text

from settings import get_engine
from utils import load_table_metadata, get_id_column


st.set_page_config(page_title="受试者档案", layout="wide")
st.title("🧬 受试者全表档案")


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
        sql = text(f"SELECT * FROM {table_name} WHERE {id_col} = :sid")
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

    st.markdown(f"共找到 **{len(subject_tables)}** 个表包含该受试者的数据。")

    # 3. 逐表展示
    for table_name, df in subject_tables.items():
        st.markdown("---")
        st.subheader(f"表：`{table_name}`  （行数：{len(df)}）")

        if len(df) <= 10:
            st.dataframe(df, use_container_width=True)
        else:
            st.caption(f"只展示前 10 行，共 {len(df)} 行。")
            show_full = st.checkbox(
                f"加载更多：显示 `{table_name}` 的全部 {len(df)} 行",
                key=f"show_full_{table_name}",
            )
            if show_full:
                st.dataframe(df, use_container_width=True)
            else:
                st.dataframe(df.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
