"""DB console entrypoint."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="DB Console", layout="wide")
st.title("DB Console")
st.markdown(
    "\n".join(
        [
            "Use the sidebar to switch pages:",
            "- Upload: import Excel/CSV into business DB.",
            "- Metadata: sync and configure tables/columns.",
        ]
    )
)
