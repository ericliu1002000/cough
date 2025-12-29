"""Deprecated compatibility layer; prefer `analysis.*` modules."""

from analysis.settings.config import get_engine
from analysis.settings.constants import OPERATORS, SUBJECT_ID_ALIASES
from analysis.repositories.metadata_repo import get_id_column, load_table_metadata
from setup_catalog.services.analysis_list_setups import (
    delete_setup_config,
    fetch_all_setups,
    fetch_all_setups_detail,
    fetch_setup_config,
    save_calculation_config,
    save_extraction_config,
    save_setup_config,
)
from analysis.repositories.sql_builder import (
    build_sql,
    format_value_for_sql,
    get_unique_values,
)

__all__ = [
    "get_engine",
    "OPERATORS",
    "SUBJECT_ID_ALIASES",
    "get_id_column",
    "load_table_metadata",
    "delete_setup_config",
    "fetch_all_setups",
    "fetch_all_setups_detail",
    "fetch_setup_config",
    "save_calculation_config",
    "save_extraction_config",
    "save_setup_config",
    "build_sql",
    "format_value_for_sql",
    "get_unique_values",
]
