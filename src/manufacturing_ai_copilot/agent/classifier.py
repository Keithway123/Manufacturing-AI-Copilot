from typing import TypedDict

KNOWLEDGE_QA = "knowledge_qa"
TOOL_REQUEST = "tool_request"
UNKNOWN = "unknown"

EQUIPMENT_SOP = "equipment_sop"
PRODUCTION_ORDER = "production_order"
QUALITY_ISSUE = "quality_issue"
IT_SUPPORT = "it_support"
WORK_ORDER_STATUS = "work_order_status"
GENERAL_KNOWLEDGE = "general_knowledge"
UNKNOWN_DOMAIN = "unknown"

KNOWLEDGE_QA_KEYWORDS = (
    "报警",
    "处理",
    "SOP",
    "点检",
    "工单",
    "MES",
    "8D",
    "质量",
    "VPN",
    "密码",
    "注塑",
    "贴片机",
    "设备",
)

WORK_ORDER_STATUS_KEYWORDS = (
    "工单状态",
    "当前状态",
    "查询状态",
)

# 每个业务域独立维护关键词，避免和“是否进入知识库问答”的判断混在一起
DOMAIN_KEYWORDS = {
    EQUIPMENT_SOP: (
        "报警",
        "SOP",
        "点检",
        "注塑",
        "贴片机",
        "设备",
        "吸嘴",
    ),
    PRODUCTION_ORDER: (
        "MES",
        "工单",
        "生产状态",
        "暂停",
        "完成",
    ),
    QUALITY_ISSUE: (
        "质量",
        "8D",
        "根因",
        "D3",
        "异常",
    ),
    IT_SUPPORT: (
        "VPN",
        "密码",
        "账号",
        "IT",
    ),
}

ROUTE_RAG_ANSWER = "rag_answer"
ROUTE_TOOL_NODE = "tool_node"
ROUTE_FALLBACK = "fallback"


class ClassificationResult(TypedDict):
    question_type: str
    domain_type: str
    route: str


def classify_question(question: str) -> ClassificationResult:
    question_type = UNKNOWN
    route = ROUTE_FALLBACK
    domain_type = UNKNOWN_DOMAIN

    for keyword in WORK_ORDER_STATUS_KEYWORDS:
        if keyword.lower() in question.lower():
            return {
                "question_type": TOOL_REQUEST,
                "domain_type": WORK_ORDER_STATUS,
                "route": ROUTE_TOOL_NODE,
            }

    for keyword in KNOWLEDGE_QA_KEYWORDS:
        if keyword.lower() in question.lower():
            question_type = KNOWLEDGE_QA
            route = ROUTE_RAG_ANSWER
            domain_type = GENERAL_KNOWLEDGE
            break

    if question_type == KNOWLEDGE_QA:
        for domain, keywords in DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in question.lower():
                    domain_type = domain
                    break

            if domain_type == domain:
                break

    return {
        "question_type": question_type,
        "domain_type": domain_type,
        "route": route,
    }
