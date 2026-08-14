from contextlib import closing
from typing import Any

from manufacturing_ai_copilot.db.connection import (
    get_connection,
    get_parameter_placeholder,
)

SELECT_WORK_ORDER_BY_ID_SQL_TEMPLATE = """
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
WHERE work_order_id = {placeholder}
"""


def get_work_order_by_id(work_order_id: str) -> dict[str, Any] | None:
    with closing(get_connection()) as connection:
        sql = SELECT_WORK_ORDER_BY_ID_SQL_TEMPLATE.format(
            placeholder=get_parameter_placeholder(connection),
        )

        row = connection.execute(
            sql,
            (work_order_id,),
        ).fetchone()

    if row is None:
        return None

    return dict(row)
