"""Project docs library page."""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import streamlit as st
from streamlit import config as st_config

from analysis.auth.session import require_login
from analysis.views.components.page_utils import (
    hide_login_sidebar_entry,
    render_sidebar_navigation,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = BASE_DIR / "static" / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_name(file_name: str) -> str:
    return Path(file_name).name


def _doc_path(file_name: str) -> Path:
    safe_name = _safe_name(file_name)
    return DOCS_DIR / safe_name


def _list_docs() -> list[Path]:
    docs = [
        path
        for path in DOCS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    ]
    docs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return docs


def _build_static_url(filename: str) -> str:
    base_path = st_config.get_option("server.baseUrlPath") or ""
    base_prefix = f"/{base_path.strip('/')}" if base_path else ""
    return f"{base_prefix}/app/static/docs/{quote(filename)}"


st.set_page_config(page_title="项目文档", layout="wide")
hide_login_sidebar_entry()
require_login()

st.title("📚 项目文档")

with st.sidebar:
    render_sidebar_navigation(active_page="project_docs")

st.subheader("上传 PDF 文档")
uploaded_files = st.file_uploader(
    "选择 PDF 文件（同名将覆盖）",
    type=["pdf"],
    accept_multiple_files=True,
)
if uploaded_files:
    if st.button("保存", type="primary"):
        for uploaded in uploaded_files:
            doc_path = _doc_path(uploaded.name)
            doc_path.write_bytes(uploaded.getvalue())
        st.success("文档已保存。")
        st.rerun()

st.divider()
st.subheader("文档列表")

docs = _list_docs()
if not docs:
    st.info("暂无文档，请先上传。")
else:
    for doc_path in docs:
        updated_at = datetime.fromtimestamp(
            doc_path.stat().st_mtime
        ).strftime("%Y-%m-%d %H:%M")
        col_name, col_time, col_open, col_delete = st.columns([5, 2, 1, 1])
        col_name.markdown(f"**{doc_path.name}**")
        col_time.caption(f"更新: {updated_at}")

        viewer_url = _build_static_url(doc_path.name)
        col_open.markdown(
            f"<a href=\"{html.escape(viewer_url, quote=True)}\" "
            "target=\"_blank\">打开</a>",
            unsafe_allow_html=True,
        )

        if col_delete.button("删除", key=f"delete_{doc_path.name}"):
            doc_path.unlink(missing_ok=True)
            st.success(f"已删除: {doc_path.name}")
            st.rerun()
