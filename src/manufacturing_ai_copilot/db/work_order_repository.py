from contextlib import closing
from typing import Any

from manufacturing_ai_copilot.db.connection import get_connection

SELECT_WORK_ORDER_BY_ID_SQL = """
SELECT 
    work_order_id,
    status,
    product,
    line,
    planned_quantity,
    completed_quantity,
    pause_reason,
    created_at,
    updated_at
FROM work_orders
WHERE work_order_id = ?
"""


def get_work_order_by_id(work_order_id: str) -> dict[str, Any] | None:
    with closing(get_connection()) as connection:
        row = connection.execute(
            SELECT_WORK_ORDER_BY_ID_SQL,
            (work_order_id,),
        ).fetchone()

    if row is None:
        return None

    return dict(row)
