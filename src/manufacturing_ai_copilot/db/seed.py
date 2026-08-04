from contextlib import closing
from manufacturing_ai_copilot.db.connection import get_connection

WORK_ORDER_SEED_DATA = [
    (
        "WO-20260727-001",
        "paused",
        "SMT Controller Board",
        "SMT-01",
        1000,
        420,
        "equipment_alarm",
    ),
    (
        "WO-20260727-002",
        "planned",
        "Power Module",
        "SMT-02",
        800,
        0,
        None,
    ),
    (
        "WO-20260727-003",
        "in_progress",
        "Sensor Board",
        "SMT-01",
        1200,
        650,
        None,
    ),
    (
        "WO-20260727-004",
        "completed",
        "Control Panel",
        "ASSEMBLY-01",
        500,
        500,
        None,
    ),
    (
        "WO-20260727-005",
        "cancelled",
        "Drive Module",
        "SMT-02",
        700,
        0,
        None,
    ),
]

INSERT_WORK_ORDER_SQL = """
INSERT INTO work_orders(
    work_order_id,
    status,
    product,
    line,
    planned_quantity,
    completed_quantity,
    pause_reason
)
VALUES(?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(work_order_id) DO NOTHING
"""


def seed_work_orders() -> int:
    # 只忽略重复工单; 状态或数量等其他约束错误仍然抛出。
    with closing(get_connection()) as connection:
        cursor = connection.executemany(
            INSERT_WORK_ORDER_SQL,
            WORK_ORDER_SEED_DATA,
        )
        connection.commit()
        return cursor.rowcount
