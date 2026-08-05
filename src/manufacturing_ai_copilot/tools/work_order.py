import re
from typing import Any

from manufacturing_ai_copilot.db.work_order_repository import (
    get_work_order_by_id,
)

TOOL_REASON_MISSING_WORK_ORDER_ID = "missing_work_order_id"
TOOL_REASON_WORK_ORDER_NOT_FOUND = "work_order_not_found"

WORK_ORDER_ID_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])WO-\d{8}-\d{3}(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def extract_work_order_id(question: str) -> str | None:
    match = WORK_ORDER_ID_PATTERN.search(question)

    if match is None:
        return None

    return match.group(0).upper()


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
            "reason": TOOL_REASON_WORK_ORDER_NOT_FOUND,
            "source": "database",
        }

    # Tool 层组装稳定契约，不直接把数据库整行暴露给graph。
    return {
        "tool_name": "query_work_order_status",
        "work_order_id": work_order["work_order_id"],
        "found": True,
        "reason": None,
        "status": work_order["status"],
        "product": work_order["product"],
        "line": work_order["line"],
        "planned_quantity": work_order["planned_quantity"],
        "completed_quantity": work_order["completed_quantity"],
        "pause_reason": work_order["pause_reason"],
        "source": "database",
    }
