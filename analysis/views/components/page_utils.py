"""Shared page-level formatting utilities."""

from urllib.parse import urlencode

import streamlit as st
from streamlit import config as st_config

from analysis.auth.session import append_auth_token


def build_page_url(
    page_name: str | None,
    params: dict[str, str] | None = None,
) -> str:
    """Build a page URL with the configured base path and query params."""
    base_path = st_config.get_option("server.baseUrlPath") or ""
    base_prefix = f"/{base_path.strip('/')}" if base_path else ""
    params = append_auth_token(params)
    if page_name:
        base_url = f"{base_prefix}/{page_name}"
    else:
        base_url = base_prefix or "/"
    if params:
        query = urlencode(params)
        return f"{base_url}?{query}"
    return base_url


def truncate_text(text: str, max_len: int) -> str:
    """Truncate text with ellipsis when it exceeds max_len."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def hide_login_sidebar_entry() -> None:
    """Hide the login page entry in the Streamlit sidebar navigation."""
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] a[href*="_login"],
        section[data-testid="stSidebar"] a[href*="login"],
        section[data-testid="stSidebar"] a[aria-label*="Login"],
        section[data-testid="stSidebar"] button[aria-label*="Login"],
        section[data-testid="stSidebar"] li:has(a[href*="_login"]),
        section[data-testid="stSidebar"] li:has(a[href*="login"]),
        section[data-testid="stSidebar"] li:has(a[aria-label*="Login"]),
        section[data-testid="stSidebar"] li:has(button[aria-label*="Login"]) {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_navigation(active_page: str | None = None) -> None:
    """Render the shared sidebar navigation links."""
    nav_items = [
        ("analysis_setups", "Analysis Setups", None),
        ("data_builder", "Dataset Builder", "data_builder"),
        ("analysis_dashboard", "Analysis Dashboard", "analysis_dashboard"),
        ("subject_profile", "Profile Book", "subject_profile"),
        ("project_docs", "Docs", "project_docs"),
        ("db_upload", "Raw Data Upload", "db_upload"),
        ("db_metadata", "Metadata Config", "db_metadata"),
        ("db_metadata_column", "Column Config", "db_metadata_column"),
    ]
    st.markdown("**Navigation**")
    query_params = append_auth_token({})
    for item_id, label, page_name in nav_items:
        if active_page == item_id:
            st.markdown(f"**{label}**")
            continue
        page_path = (
            f"pages/{page_name}.py" if page_name else "analysis_setups.py"
        )
        st.page_link(
            page_path,
            label=label,
            query_params=query_params or None,
        )
    st.markdown("---")
