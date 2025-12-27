"""Streamlit login page."""

import streamlit as st

from analysis.auth.session import (
    authenticate,
    mark_authenticated,
    pop_notice,
    is_session_valid,
    sync_auth_from_query,
    touch_session,
)


def main() -> None:
    """Render the login page and handle authentication."""
    st.set_page_config(page_title="登录", layout="centered")
    st.title("🔐 登录")

    if is_session_valid():
        touch_session()
        st.switch_page("analysis_setups.py")
        st.stop()
    if sync_auth_from_query():
        st.switch_page("analysis_setups.py")
        st.stop()

    notice = pop_notice()
    if notice:
        st.warning(notice)

    with st.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submitted = st.form_submit_button("登录")

    if submitted:
        if authenticate(username, password):
            mark_authenticated(username)
            st.success("登录成功")
            st.switch_page("analysis_setups.py")
            st.stop()
        st.error("用户名或密码错误")


if __name__ == "__main__":
    main()
