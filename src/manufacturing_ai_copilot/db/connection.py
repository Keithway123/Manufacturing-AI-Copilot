import sqlite3
from pathlib import Path

from manufacturing_ai_copilot.core.config import DATABASE_URL

SQLITE_URL_PREFIX = "sqlite:///"


def get_sqlite_path(database_url: str = DATABASE_URL) -> Path:
    if not database_url.startswith(SQLITE_URL_PREFIX):
        raise ValueError("DATABASE_URL must use sqlite:/// format")

    return Path(database_url.removeprefix(SQLITE_URL_PREFIX))


def get_connection(database_url: str = DATABASE_URL) -> sqlite3.Connection:
    database_path = get_sqlite_path(database_url)

    database_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    return connection
