from manufacturing_ai_copilot.db import work_order_repository
from manufacturing_ai_copilot.db.connection import get_connection
from manufacturing_ai_copilot.db.schema import CREATE_WORK_ORDERS_TABLE_SQL


def prepare_test_database(tmp_path, monkeypatch) -> None:
    # tmp_path由pytest提供，测试结束后自动清理。
    database_path = tmp_path / "test_manufacturing.db"
    database_url = f"sqlite:///{database_path.as_posix()}"

    connection = get_connection(database_url)
    connection.execute(CREATE_WORK_ORDERS_TABLE_SQL)

    # 固定时间值，让测试结果稳定，不依赖执行时刻。
    connection.execute(
        """
        INSERT INTO work_orders(
            work_order_id,
            status,
            product,
            line,
            planned_quantity,
            completed_quantity,
            pause_reason,
            created_at,
            updated_at
    )
    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            "WO-TEST-001",
            "paused",
            "Test Product",
            "TEST-LINE",
            100,
            40,
            "equipment_alarm",
            "2026-08-04 10:00:00",
            "2026-08-04 10:00:00",
        ),
    )
    connection.commit()
    connection.close()

    # Repository 已经导入了get_connection,因此替换它实际使用的名字。
    monkeypatch.setattr(
        work_order_repository,
        "get_connection",
        lambda: get_connection(database_url),
    )


def test_get_work_order_by_id_returns_work_order(tmp_path, monkeypatch):
    prepare_test_database(tmp_path, monkeypatch)

    result = work_order_repository.get_work_order_by_id("WO-TEST-001")

    assert result == {
        "work_order_id": "WO-TEST-001",
        "status": "paused",
        "product": "Test Product",
        "line": "TEST-LINE",
        "planned_quantity": 100,
        "completed_quantity": 40,
        "pause_reason": "equipment_alarm",
        "created_at": "2026-08-04 10:00:00",
        "updated_at": "2026-08-04 10:00:00",
    }


def test_get_work_order_by_id_returns_none_when_missing(tmp_path, monkeypatch):
    prepare_test_database(tmp_path, monkeypatch)

    result = work_order_repository.get_work_order_by_id("WO-NOT-FOUND")

    assert result is None
