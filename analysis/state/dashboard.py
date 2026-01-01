"""Session state helpers for the analysis dashboard."""

import streamlit as st


def reset_dashboard_state() -> None:
    """Clear cached UI state before loading a saved config."""
    reset_keys = [
        "calc_note_input",
        "bl_subj_ui",
        "bl_visit_ui",
        "bl_val_ui",
        "bl_targets_ui",
        "ex_f",
        "ex_v",
        "pivot_row_order_selected",
        "pivot_row_order_up",
        "pivot_row_order_down",
        "pivot_agg_axis_ui",
        "boxplot_visible_cols",
    ]
    reset_prefixes = [
        "pivot_row_order_selected_",
        "pivot_row_order_up_",
        "pivot_row_order_down_",
        "pivot_col_order_selected_",
        "pivot_col_order_up_",
        "pivot_col_order_down_",
        "ex_field_",
        "ex_op_",
        "ex_vals_",
        "ex_value_",
        "agg_group_",
        "agg_col_",
        "agg_func_",
        "agg_name_",
        "agg_del_",
    ]
    for key in reset_keys:
        st.session_state.pop(key, None)
    for key in list(st.session_state.keys()):
        if any(key.startswith(prefix) for prefix in reset_prefixes):
            st.session_state.pop(key, None)
