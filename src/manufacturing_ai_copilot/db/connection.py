import sqlite3
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from manufacturing_ai_copilot.core.config import DATABASE_URL

SQLITE_URL_PREFIX = "sqlite:///"
POSTGRESQL_URL_PREFIX = "postgresql://"

# 两种连接拥有相近的 execute/commit/close 接口，供上层统一调用。
DatabaseConnection = sqlite3.Connection | psycopg.Connection[dict[str, Any]]


def get_parameter_placeholder(
    connection: DatabaseConnection,
) -> str:
    if isinstance(connection, sqlite3.Connection):
        return "?"
    if isinstance(connection, psycopg.Connection):
        return "%s"

    raise TypeError("Unsupported database connection type")


def get_sqlite_path(database_url: str = DATABASE_URL) -> Path:
    if not database_url.startswith(SQLITE_URL_PREFIX):
        raise ValueError("DATABASE_URL must use sqlite:/// format")

    return Path(database_url.removeprefix(SQLITE_URL_PREFIX))


def get_connection(database_url: str = DATABASE_URL) -> DatabaseConnection:
    if database_url.startswith(SQLITE_URL_PREFIX):
        database_path = get_sqlite_path(database_url)
        database_path.parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(database_path)
        connection.row_factory = sqlite3.Row
        return connection

    if database_url.startswith(POSTGRESQL_URL_PREFIX):
        return psycopg.connect(
            database_url,
            row_factory=dict_row,
        )

    raise ValueError("DATABASE_URL must use sqlite:/// or postgresql:// format")
