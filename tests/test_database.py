from contextlib import closing
from pathlib import Path

import sqlite3
import pytest

from manufacturing_ai_copilot.db import connection, schema, seed
from manufacturing_ai_copilot.db.connection import (
    get_connection,
    get_sqlite_path,
)
from manufacturing_ai_copilot.db.errors import DatabaseUnavailableError


@pytest.fixture
def temporary_database_url(tmp_path, monkeypatch) -> str:
    database_path = tmp_path / "test_manufacturing.db"
    database_url = f"sqlite:///{database_path.as_posix()}"

    def temporary_get_connection():
        # 每次建立新连接，但都指向当前测试的同一个数据库文件。
        return get_connection(database_url)

    # schema.py 和 seed.py 都直接导入了get_connection,
    # 因此要替换他们实际使用的模块变量。
    monkeypatch.setattr(
        schema,
        "get_connection",
        temporary_get_connection,
    )
    monkeypatch.setattr(
        seed,
        "get_connection",
        temporary_get_connection,
    )

    return database_url


def test_get_sqlite_path_returns_path():
    result = get_sqlite_path("sqlite:///tmp/test_manufacturing.db")

    assert result == Path("tmp/test_manufacturing.db")


def test_get_sqlite_path_rejects_non_sqlite_url():
    # 非SQLite URL属于配置错误 应该明确抛出异常。
    with pytest.raises(
        ValueError,
        match="DATABASE_URL must use sqlite",
    ):
        get_sqlite_path("postgresql://localhost/manufacturing")


def test_initialize_database_is_idempotent(
    temporary_database_url,
):
    schema.initialize_database()
    schema.initialize_database()

    with closing(get_connection(temporary_database_url)) as connection:
        row = connection.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
             AND name = 'work_orders'
""").fetchone()

    assert row is not None
    assert row["name"] == "work_orders"


def test_seed_work_orders_is_idempotent(
    temporary_database_url,
):
    schema.initialize_database()

    first_inserted = seed.seed_work_orders()
    second_inserted = seed.seed_work_orders()

    with closing(get_connection(temporary_database_url)) as connection:
        row = connection.execute("SELECT COUNT(*) AS count FROM work_orders").fetchone()

    assert first_inserted == 5
    assert second_inserted == 0
    assert row["count"] == 5


def test_work_orders_rejects_invalid_status(temporary_database_url):
    # 插入非法状态 running，验证 status CHECK
    # 先创建表，否则无法测试表约束。
    schema.initialize_database()

    with closing(get_connection(temporary_database_url)) as connection:
        # 这里直接执行 INSERT，是为了测试数据库表自己的 CHECK 约束。
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO work_orders (
                    work_order_id,
                    status,
                    product,
                    line,
                    planned_quantity,
                    completed_quantity
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "WO-INVALID-STATUS",
                    "running",
                    "Test Product",
                    "TEST-01",
                    100,
                    0,
                ),
            )


def test_work_orders_rejects_negative_quantity(temporary_database_url):
    # 插入负数 planned_quantity，验证数量约束
    # 先创建表，否则无法测试表约束。
    schema.initialize_database()

    with closing(get_connection(temporary_database_url)) as connection:
        # planned_quantity 不能为负数，这属于数据库层的数据完整性保护。
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO work_orders (
                    work_order_id,
                    status,
                    product,
                    line,
                    planned_quantity,
                    completed_quantity
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "WO-NEGATIVE-QTY",
                    "planned",
                    "Test Product",
                    "TEST-01",
                    -1,
                    0,
                ),
            )


def test_get_connection_converts_sqlite_operational_error(
    tmp_path,
    monkeypatch,
):
    database_url = f"sqlite:///{(tmp_path / 'unavailable.db').as_posix()}"

    def fake_sqlite_connect(database_path):
        # 模拟SQLite文件无法打开。
        raise sqlite3.OperationalError("unable to open database file")

    # 替换connection.py 实际查找和调用的connect。
    monkeypatch.setattr(
        connection.sqlite3,
        "connect",
        fake_sqlite_connect,
    )

    with pytest.raises(
        DatabaseUnavailableError,
        match="Database connection failed",
    ) as exc_info:
        get_connection(database_url)

    # 验证 raise ... from exc 保留了原始驱动异常。
    assert isinstance(
        exc_info.value.__cause__,
        sqlite3.OperationalError,
    )


def test_get_connection_converts_postgresql_operational_error(
    monkeypatch,
):
    def fake_psycopg_connect(database_url, **kwargs):
        # 模拟 PostgreSQL 停止或拒绝连接。
        raise connection.psycopg.OperationalError("connection refused")

    monkeypatch.setattr(
        connection.psycopg,
        "connect",
        fake_psycopg_connect,
    )

    with pytest.raises(
        DatabaseUnavailableError,
        match="Database connection failed",
    ) as exc_info:
        get_connection("postgresql://user:password@postgres:5432/manufacturing")

    # 项目异常应保留底层 Psycopg 异常。
    assert isinstance(
        exc_info.value.__cause__,
        connection.psycopg.OperationalError,
    )
