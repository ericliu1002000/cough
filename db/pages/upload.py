"""业务数据上传页面（单文件）。"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from db.services.db_config import get_business_db_config
from db.services.upload import FileFormatError, upload_csv, upload_excel

load_dotenv(BASE_DIR / ".env")

def _parse_list_env(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if raw is None:
        return list(default)
    items = [part.strip() for part in raw.split(",") if part.strip()]
    return items


def _parse_int_list_env(name: str, default: List[int]) -> List[int]:
    raw = os.getenv(name)
    if raw is None:
        return list(default)
    items: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            items.append(int(part))
        except ValueError:
            continue
    return items


def _get_server_url(config: Dict[str, str]) -> URL:
    return URL.create(
        drivername="mysql+pymysql",
        username=config["user"],
        password=config["password"] or None,
        host=config["host"],
        port=int(config["port"]),
        database=None,
    )


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
    safe_db_name = config["database"].replace("`", "``")
    engine = create_engine(_get_server_url(config), future=True)
    create_sql = text(
        f"CREATE DATABASE IF NOT EXISTS `{safe_db_name}` "
        "DEFAULT CHARACTER SET utf8mb4 "
        "COLLATE utf8mb4_unicode_ci"
    )
    with engine.connect() as conn:
        conn.execute(create_sql)
        conn.commit()
    engine.dispose()


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
excel_skip_rows = _parse_int_list_env("SKIP_ROW_NUMBER", [2])
csv_skip_rows = _parse_int_list_env("CSV_SKIP_ROW_NUMBER", [])

st.warning("注意：如果表已存在，将会全量替换数据库中的数据。")

st.markdown("### 导入规则")
st.markdown(
    "\n".join(
        [
            "- 单文件上传，Excel 可包含多个 Sheet。",
            "- 表名规则：Excel 使用 Sheet 名称；CSV 使用文件名（不含扩展名）。",
            "- 禁止 Sheet 名包含 'sheet'（不区分大小写），需改名后上传。",
            f"- 跳过 Sheet：{', '.join(skip_sheets) if skip_sheets else '无'}",
            (
                "- Excel 跳过行号（表格行号，含表头）："
                f"{', '.join(map(str, excel_skip_rows)) or '无'}"
            ),
            (
                "- CSV 跳过行号（表格行号，含表头）："
                f"{', '.join(map(str, csv_skip_rows)) or '无'}"
            ),
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
            engine = _get_business_engine(config)
            with st.spinner("正在导入..."):
                if file_ext in {".xlsx", ".xls"}:
                    logs = upload_excel(
                        uploaded_file,
                        engine,
                        skip_sheets,
                        excel_skip_rows,
                    )
                elif file_ext == ".csv":
                    logs = upload_csv(
                        uploaded_file,
                        engine,
                        csv_skip_rows,
                    )
                else:
                    raise FileFormatError("文件格式不正确")
            st.success("导入完成。")
            if logs:
                st.code("\n".join(logs))
        except FileFormatError:
            st.error("文件格式不正确")
        except Exception as exc:
            st.error(f"导入失败: {exc}")
