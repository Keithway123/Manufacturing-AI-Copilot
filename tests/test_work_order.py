from manufacturing_ai_copilot.tools.work_order import extract_work_order_id


def test_extract_work_order_id_from_question():
    result = extract_work_order_id("查询工单WO-20260727-001当前状态")

    assert result == "WO-20260727-001"


def test_extract_work_order_id_normalizes_lowercase():
    result = extract_work_order_id("查询 wo-20260727-002 当前状态")

    assert result == "WO-20260727-002"


def test_extract_work_order_id_returns_none_when_missing():
    result = extract_work_order_id("查询工单当前状态")

    assert result is None


def test_extract_work_order_id_rejects_longer_identifier():
    # 末尾多一位数字时，不能错误截取前面的合法部分。
    result = extract_work_order_id("查询工单 WO-20260727-0019 当前状态")

    assert result is None
