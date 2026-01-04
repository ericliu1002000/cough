"""Wrapper page for DB metadata."""

import streamlit as st

import db.pages.metadata as _page
from analysis.views.components.page_utils import (
    hide_login_sidebar_entry,
    render_sidebar_navigation,
)

hide_login_sidebar_entry()
with st.sidebar:
    render_sidebar_navigation(active_page="db_metadata")
