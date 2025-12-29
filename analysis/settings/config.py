"""App configuration and database connection helpers."""

import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
# ========================
# 环境与路径相关配置
# ========================

# 项目根目录（当前文件所在目录）
# 以后所有相对路径（如 data 目录、.env）都基于这个目录来计算
BASE_DIR = Path(__file__).resolve().parents[2]

# 从项目根目录加载 .env 中的环境变量
# 典型内容示例：
#   MYSQL_USER=root
#   MYSQL_PASSWORD=***
#   MYSQL_DATABASE=analysis_db
#   MYSQL_PORT=3307
#   MYSQL_HOST=127.0.0.1
load_dotenv(BASE_DIR / ".env")

# 数据目录：
# - 默认：项目根目录下的 data
# - 如需自定义，可在 .env 中设置 DATA_DIR=/your/path
DATA_DIR = os.getenv("DATA_DIR", str(BASE_DIR / "data"))

# ========================
# 数据库连接相关配置
# ========================

# 数据库基础连接信息，全部来自 .env 中的 MYSQL_* 变量
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

# 是否在控制台输出 SQL 语句，方便调试
# - 在 .env 中设置 ENGINE_ECHO=true 即可打开
ENGINE_ECHO = os.getenv("ENGINE_ECHO", "false").lower() == "true"


def _get_server_url() -> URL:
    """
    构造「不指定数据库」的 MySQL 连接 URL。

    使用场景：
    - 在第一次连接时，目标库可能还不存在；
    - 因此先连接到「服务器层面」（不指定 database），
      再执行 CREATE DATABASE 语句创建目标库。
    """
    return URL.create(
        drivername="mysql+pymysql",
        username=MYSQL_USER,
        password=MYSQL_PASSWORD or None,
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        database=None,
    )


def _get_db_url() -> URL:
    """
    构造指定数据库名的 MySQL 连接 URL，用于后续正常读写。

    注意：
    - 如果 .env 中没有配置 MYSQL_DATABASE，将直接抛出异常，
      防止在未知数据库名的情况下误操作。
    """
    if not MYSQL_DATABASE:
        raise ValueError("环境变量 MYSQL_DATABASE 不能为空")

    return URL.create(
        drivername="mysql+pymysql",
        username=MYSQL_USER,
        password=MYSQL_PASSWORD or None,
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        database=MYSQL_DATABASE,
    )


def _get_server_url_for_config(config: Dict[str, str]) -> URL:
    """基于传入配置构造「不指定数据库」的连接 URL。"""
    return URL.create(
        drivername="mysql+pymysql",
        username=config["user"],
        password=config["password"] or None,
        host=config["host"],
        port=int(config["port"]),
        database=None,
    )


def _get_db_url_for_config(config: Dict[str, str]) -> URL:
    """基于传入配置构造带数据库名的连接 URL。"""
    return URL.create(
        drivername="mysql+pymysql",
        username=config["user"],
        password=config["password"] or None,
        host=config["host"],
        port=int(config["port"]),
        database=config["database"],
    )


def ensure_database_exists() -> None:
    """
    确保目标数据库存在：
    - 若已存在：不做任何修改；
    - 若不存在：自动执行 CREATE DATABASE 创建。

    设计原因：
    - main.py 在真正写入数据前，只需关心业务逻辑；
    - 数据库库级别的准备工作统一由本函数负责。
    """
    if not MYSQL_DATABASE:
        raise ValueError("环境变量 MYSQL_DATABASE 不能为空")

    # 简单转义反引号，避免语法问题
    safe_db_name = MYSQL_DATABASE.replace("`", "``")

    server_url = _get_server_url()
    engine = create_engine(server_url, echo=ENGINE_ECHO, future=True)

    create_db_sql = text(
        f"CREATE DATABASE IF NOT EXISTS `{safe_db_name}` "
        "DEFAULT CHARACTER SET utf8mb4 "
        "COLLATE utf8mb4_unicode_ci"
    )

    print(f"[settings] 确保数据库存在: {MYSQL_DATABASE}")
    with engine.connect() as conn:
        conn.execute(create_db_sql)
        conn.commit()

    engine.dispose()


def get_engine():
    """
    返回一个可直接用于读写的 SQLAlchemy Engine。

    内部步骤：
    1. 调用 ensure_database_exists()，确保目标数据库已经创建；
    2. 使用带数据库名的 URL 创建 Engine；
    3. 打印当前使用的连接信息，便于调试。
    """
    ensure_database_exists()
    db_url = _get_db_url()
    engine = create_engine(db_url, echo=ENGINE_ECHO, future=True)
    return engine


def ensure_database_exists_for_config(config: Dict[str, str]) -> None:
    """根据传入配置确保数据库存在。"""
    safe_db_name = config["database"].replace("`", "``")
    server_url = _get_server_url_for_config(config)
    engine = create_engine(server_url, echo=ENGINE_ECHO, future=True)
    create_db_sql = text(
        f"CREATE DATABASE IF NOT EXISTS `{safe_db_name}` "
        "DEFAULT CHARACTER SET utf8mb4 "
        "COLLATE utf8mb4_unicode_ci"
    )
    print(f"[settings] 确保数据库存在: {config['database']}")
    with engine.connect() as conn:
        conn.execute(create_db_sql)
        conn.commit()
    engine.dispose()


def get_system_engine():
    """系统库连接（基于 MYSQL_* 环境变量）。"""
    return get_engine()


def get_business_engine():
    """业务库连接（基于 CURRENT_BUSINESS_CODE 对应的配置）。"""
    from db.services.db_config import get_business_db_config

    config = get_business_db_config()
    ensure_database_exists_for_config(config)
    db_url = _get_db_url_for_config(config)
    return create_engine(db_url, echo=ENGINE_ECHO, future=True)
