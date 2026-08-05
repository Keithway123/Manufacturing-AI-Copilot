from typing import Any

from manufacturing_ai_copilot.db.work_order_repository import (
    get_work_order_by_id,
)


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


def query_work_order_status(work_order_id: str) -> dict[str, Any]:
    work_order = get_work_order_by_id(work_order_id)

    # 未找到也是正常业务结果，不属于程序异常。
    if work_order is None:
        return {
            "tool_name": "query_work_order_status",
            "work_order_id": work_order_id,
            "found": False,
            "source": "database",
        }

    # Tool 层组装稳定契约，不直接把数据库整行暴露给graph。
    return {
        "tool_name": "query_work_order_status",
        "work_order_id": work_order["work_order_id"],
        "found": True,
        "status": work_order["status"],
        "product": work_order["product"],
        "line": work_order["line"],
        "planned_quantity": work_order["planned_quantity"],
        "completed_quantity": work_order["completed_quantity"],
        "pause_reason": work_order["pause_reason"],
        "source": "database",
    }
