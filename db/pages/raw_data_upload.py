"""业务数据上传页面（单文件）。"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from db.services.db_config import get_business_db_config
from db.services.init_system_db import init_system_db
from db.services.metadata import (
    sync_business_metadata,
    update_business_column_display_names,
)
from db.services.upload import FileFormatError, upload_csv, upload_excel
from analysis.settings.config import ensure_database_exists_for_config

load_dotenv(BASE_DIR / ".env")

def _parse_list_env(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if raw is None:
        return list(default)
    items = [part.strip() for part in raw.split(",") if part.strip()]
    return items


def _parse_display_row_env(name: str) -> tuple[Optional[int], str]:
    raw = os.getenv(name)
    if raw is None:
        return None, "自动识别"
    text = raw.strip()
    if not text:
        return None, "自动识别"
    try:
        value = int(text)
    except ValueError:
        return None, f"自动识别（无效配置: {text}）"
    if value <= 0:
        return None, "自动识别"
    return value, str(value)


def _get_db_url(config: Dict[str, str]) -> URL:
    return URL.create(
        drivername="mysql+pymysql",
        username=config["user"],
        password=config["password"] or None,
        host=config["host"],
        port=int(config["port"]),
        database=config["database"],
    )


def _ensure_business_database_exists(config: Dict[str, str]) -> None:
    ensure_database_exists_for_config(config)


def _get_business_engine(config: Dict[str, str]):
    _ensure_business_database_exists(config)
    return create_engine(_get_db_url(config), future=True)


st.set_page_config(page_title="业务数据上传", layout="wide")
st.title("📥 业务数据上传")

try:
    config = get_business_db_config()
except ValueError as exc:
    st.error(str(exc))
    st.stop()
skip_sheets = _parse_list_env(
    "SKIP_SHEETS", ["Event workflow", "Database Structure", "Cover Page"]
)
excel_display_row, excel_display_label = _parse_display_row_env(
    "DISPLAY_NAME_ROW"
)
csv_display_row, csv_display_label = _parse_display_row_env(
    "CSV_DISPLAY_NAME_ROW"
)

st.warning("注意：如果表已存在，将会全量替换数据库中的数据。")

st.markdown("### 导入规则")
st.markdown(
    "\n".join(
        [
            "- 单文件上传，Excel 可包含多个 Sheet。",
            "- 表名规则：Excel 使用 Sheet 名称；CSV 使用文件名（不含扩展名）。",
            "- 禁止 Sheet 名包含 'sheet'（不区分大小写），需改名后上传。",
            f"- 跳过 Sheet：{', '.join(skip_sheets) if skip_sheets else '无'}",
            f"- Excel 显示名行配置：{excel_display_label}",
            f"- CSV 显示名行配置：{csv_display_label}",
            "- 自动识别显示名行时，默认对比第2行与第3行的文本占比。",
            "- 显示名写入规则：仅当系统库 display_name 为空时写入。",
            "- 导入前后会自动同步元数据。",
            "- 列名强校验：不允许重复列名、空列名或 Unnamed 列。",
        ]
    )
)

st.markdown("### 当前业务库")
st.markdown(
    "\n".join(
        [
            f"- CURRENT_BUSINESS_CODE: `{config['code']}`",
            f"- 数据库名: `{config['database']}`",
            f"- 连接地址: `{config['host']}:{config['port']}`",
        ]
    )
)

uploaded_file = st.file_uploader(
    "上传 Excel 或 CSV（单文件）",
    type=["xlsx", "xls", "csv"],
    accept_multiple_files=False,
)

if uploaded_file is not None:
    file_ext = Path(uploaded_file.name).suffix.lower()

    if st.button("开始导入", type="primary"):
        try:
            init_system_db()
            logs: List[str] = []
            _ensure_business_database_exists(config)
            try:
                with st.spinner("导入前同步元数据..."):
                    pre_sync = sync_business_metadata()
                logs.append(
                    "导入前同步："
                    f"表/视图 {pre_sync['objects_scanned']}，"
                    f"列 {pre_sync['columns_scanned']}"
                )
            except Exception as exc:
                logs.append(f"导入前同步失败: {exc}")
            engine = _get_business_engine(config)
            with st.spinner("正在导入..."):
                if file_ext in {".xlsx", ".xls"}:
                    import_logs, display_maps = upload_excel(
                        uploaded_file,
                        engine,
                        skip_sheets,
                        excel_display_row,
                    )
                elif file_ext == ".csv":
                    import_logs, display_maps = upload_csv(
                        uploaded_file,
                        engine,
                        csv_display_row,
                    )
                else:
                    raise FileFormatError("文件格式不正确")
            logs.extend(import_logs)
            st.success("导入完成。")
            try:
                with st.spinner("导入后同步元数据..."):
                    post_sync = sync_business_metadata()
                logs.append(
                    "导入后同步："
                    f"表/视图 {post_sync['objects_scanned']}，"
                    f"列 {post_sync['columns_scanned']}"
                )
            except Exception as exc:
                logs.append(f"导入后同步失败: {exc}")

            if display_maps:
                total_updates = 0
                for table_name, display_map in display_maps.items():
                    updated = update_business_column_display_names(
                        config["code"],
                        table_name,
                        display_map,
                        override=False,
                    )
                    total_updates += updated
                    logs.append(
                        f"显示名写入：{table_name} "
                        f"更新 {updated}/{len(display_map)}"
                    )
                logs.append(f"显示名写入完成，共更新 {total_updates} 个。")
            else:
                logs.append("显示名写入：无可用映射。")

            if logs:
                st.code("\n".join(logs))
        except FileFormatError:
            st.error("文件格式不正确")
        except Exception as exc:
            st.error(f"导入失败: {exc}")
