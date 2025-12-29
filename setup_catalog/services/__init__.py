"""系统配置服务集合。"""

from setup_catalog.services.analysis_list_setups import (
    delete_setup_config,
    fetch_all_setups,
    fetch_all_setups_detail,
    fetch_setup_config,
    save_calculation_config,
    save_extraction_config,
    save_setup_config,
)

__all__ = [
    "delete_setup_config",
    "fetch_all_setups",
    "fetch_all_setups_detail",
    "fetch_setup_config",
    "save_calculation_config",
    "save_extraction_config",
    "save_setup_config",
]
