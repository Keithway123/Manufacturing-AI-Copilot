from typing import Any


def query_work_order_status_stub(work_order_id: str) -> dict[str, Any]:

    return {
        "tool_name": "query_work_order_status",
        "work_order_id": work_order_id,
        "status": "paused",
        "product": "SMT Controller Board",
        "line": "SMT-01",
        "planned_quantity": 1000,
        "completed_quantity": 420,
        "source": "stub",
    }
