from contextlib import closing
from manufacturing_ai_copilot.db.connection import get_connection

CREATE_WORK_ORDERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS work_orders(
    work_order_id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK(
        status IN('planned', 'in_progress', 'paused', 'completed', 'cancelled')
        ),
        product TEXT NOT NULL,
        line TEXT NOT NULL,
        planned_quantity INTEGER NOT NULL CHECK(planned_quantity >= 0),
        completed_quantity INTEGER NOT NULL DEFAULT 0 CHECK(completed_quantity >= 0),
        pause_reason TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


def initialize_database() -> None:
    with closing(get_connection()) as connection:
        connection.execute(CREATE_WORK_ORDERS_TABLE_SQL)
        connection.commit()
