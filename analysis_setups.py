"""Streamlit setup list page."""

from __future__ import annotations

import html

import streamlit as st

from analysis.auth.session import require_login
from analysis.repositories.setup_repo import (
    fetch_all_setups_detail,
    fetch_setup_config,
    save_calculation_config,
    save_extraction_config,
)
from analysis.views.components.page_utils import build_page_url, truncate_text

st.set_page_config(page_title="分析配置", layout="wide")
require_login()

st.markdown(
    """
    <style>
    body, .stApp {
        background: #f7fbf6;
        font-family: "PingFang SC", "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif;
    }
    section.main > div.block-container {
        padding-top: 1.5rem;
    }
    .setup-grid {
        display: grid;
        grid-template-columns: repeat(6, minmax(0, 1fr));
        gap: 16px;
        margin-top: 12px;
    }
    .setup-card {
        display: block;
        padding: 14px 16px;
        border: 1px solid #d6ead7;
        border-radius: 12px;
        background: linear-gradient(135deg, #f3fbf2, #ffffff);
        text-decoration: none;
        color: #243828;
        transition: box-shadow 0.2s ease, border-color 0.2s ease,
            transform 0.2s ease;
        position: relative;
    }
    .setup-card:hover {
        border-color: #8cc99b;
        box-shadow: 0 10px 20px rgba(46, 88, 64, 0.12);
        transform: translateY(-2px);
    }
    .setup-name {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 8px;
        line-height: 1.3;
    }
    .setup-link {
        color: inherit;
        text-decoration: none;
    }
    .setup-link:hover {
        text-decoration: underline;
    }
    .setup-meta {
        font-size: 14px;
        color: #4e6b55;
        line-height: 1.5;
        min-height: 36px;
    }
    .setup-meta span {
        display: block;
    }
    div[data-testid="stExpander"] details {
        background: #f4fbf3;
        border: 1px solid #d6ead7;
        border-radius: 12px;
        padding: 4px 8px;
    }
    div[data-testid="stExpander"] summary {
        font-size: 13px;
        color: #2f5f3a;
    }
    .ops-panel {
        display: flex;
        justify-content: flex-end;
    }
    .search-panel {
        max-width: 520px;
    }
    .suggestions {
        margin-top: 8px;
        border: 1px solid #d6ead7;
        border-radius: 10px;
        background: #ffffff;
        box-shadow: 0 8px 18px rgba(46, 88, 64, 0.08);
        padding: 6px 0;
    }
    .suggest-item {
        display: block;
        padding: 8px 12px;
        color: #1e3b27;
        text-decoration: none;
    }
    .suggest-item:hover {
        background: #f0f8f1;
    }
    .suggest-name {
        font-size: 14px;
        font-weight: 600;
        line-height: 1.3;
    }
    .suggest-meta {
        font-size: 12px;
        color: #6a8372;
        line-height: 1.3;
    }
    .suggest-empty {
        padding: 10px 12px;
        color: #7a8f7f;
        font-size: 13px;
    }
    @media (max-width: 1200px) {
        .setup-grid {
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }
    }
    @media (max-width: 900px) {
        .setup-grid {
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }
    }
    @media (max-width: 600px) {
        .setup-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 420px) {
        .setup-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

setups = fetch_all_setups_detail()
setup_names = [row["setup_name"] for row in setups]


def matches_query(row: dict[str, str], query: str) -> bool:
    """Return True when any searchable fields match the query."""
    if not query:
        return True
    name = str(row.get("setup_name", "")).lower()
    desc = str(row.get("description", "")).lower()
    note = str(row.get("note", "")).lower()
    return query in name or query in desc or query in note


top_left, top_right = st.columns([6, 2])
with top_left:
    st.markdown("<div class='search-panel'>", unsafe_allow_html=True)
    search_text = st.text_input(
        "搜索",
        placeholder="搜索",
        label_visibility="collapsed",
        key="setup_search",
    )
    st.markdown("</div>", unsafe_allow_html=True)

query = search_text.strip().lower()
filtered_setups = [row for row in setups if matches_query(row, query)]

with top_left:
    if query:
        suggest_rows = filtered_setups[:10]
        if suggest_rows:
            suggestions = ["<div class='suggestions'>"]
            for row in suggest_rows:
                setup_name = str(row.get("setup_name", ""))
                description = str(row.get("description", "") or "")
                note = str(row.get("note", "") or "")
                desc_short = truncate_text(description, 46) or "无描述"
                note_short = truncate_text(note, 46) or "无备注"
                detail_url = build_page_url(
                    "analysis_dashboard", {"setup_name": setup_name}
                )
                suggestions.append(
                    "<a class='suggest-item' "
                    f"href='{html.escape(detail_url, quote=True)}' "
                    "target='_blank'>"
                    f"<div class='suggest-name'>{html.escape(setup_name)}</div>"
                    "<div class='suggest-meta'>"
                    f"描述: {html.escape(desc_short)} | "
                    f"备注: {html.escape(note_short)}"
                    "</div></a>"
                )
            suggestions.append("</div>")
            st.markdown("".join(suggestions), unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='suggestions'><div class='suggest-empty'>"
                "暂无匹配结果</div></div>",
                unsafe_allow_html=True,
            )
with top_right:
    st.markdown("<div class='ops-panel'>", unsafe_allow_html=True)
    with st.expander("复制配置", expanded=False):
        source_options = [row["setup_name"] for row in filtered_setups]
        if source_options:
            source_name = st.selectbox("来源", source_options)
        else:
            source_name = None
            st.selectbox("来源", ["暂无可选配置"], disabled=True)

        new_name = st.text_input(
            "新配置名称",
            placeholder="新名称",
        )
        new_description = st.text_input(
            "新描述",
            placeholder="描述",
        )
        do_copy = st.button("复制配置", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if do_copy:
    new_name = new_name.strip()
    new_description = new_description.strip()
    if not source_name:
        st.error("请先选择复制来源。")
    elif not new_name:
        st.error("新配置名称不能为空。")
    elif not new_description:
        st.error("新描述不能为空。")
    elif new_name in setup_names:
        st.error("新配置名称已存在，请更换名称。")
    else:
        cfg = fetch_setup_config(source_name) or {}
        extraction_cfg = cfg.get("extraction") or {}
        calculation_cfg = cfg.get("calculation") or {}
        save_extraction_config(new_name, new_description, extraction_cfg)
        save_calculation_config(new_name, calculation_cfg)
        st.success("复制完成。")
        st.rerun()

st.caption(f"共 {len(filtered_setups)} 条结果。")

if not filtered_setups:
    st.info("暂无匹配的配置。")
else:
    cards = ["<div class='setup-grid'>"]
    for row in filtered_setups:
        setup_name = str(row.get("setup_name", ""))
        description = str(row.get("description", "") or "")
        note = str(row.get("note", "") or "")
        desc_short = truncate_text(description, 56) or "无描述"
        note_short = truncate_text(note, 56) or "无备注"

        hover_text = " | ".join(
            [
                f"描述: {description}" if description else "描述: 无",
                f"备注: {note}" if note else "备注: 无",
            ]
        )
        detail_url = build_page_url(
            "analysis_dashboard", {"setup_name": setup_name}
        )
        cards.append(
            "<div class='setup-card' "
            f"title='{html.escape(hover_text)}'>"
            "<div class='setup-name'>"
            f"<a class='setup-link' href='{html.escape(detail_url, quote=True)}' "
            "target='_blank'>"
            f"{html.escape(setup_name)}</a></div>"
            "<div class='setup-meta'>"
            f"<span>描述: {html.escape(desc_short)}</span>"
            f"<span>备注: {html.escape(note_short)}</span>"
            "</div>"
            "</div>"
        )
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)
