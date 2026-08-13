from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from manufacturing_ai_copilot.core.config import MIN_RETRIEVAL_SCORE, NO_ANSWER_MESSAGE
from manufacturing_ai_copilot.rag.query_engine import chat_with_qdrant_retrieval

from manufacturing_ai_copilot.agent.classifier import (
    ROUTE_RAG_ANSWER,
    ROUTE_TOOL_NODE,
    ROUTE_FALLBACK,
    UNKNOWN,
    UNKNOWN_DOMAIN,
    classify_question,
)
from manufacturing_ai_copilot.tools.work_order import (
    query_work_order_status,
    TOOL_REASON_MISSING_WORK_ORDER_ID,
    extract_work_order_id,
)

ANSWER_QUALITY_GROUNDED = "grounded"
ANSWER_QUALITY_WEAK = "weak"
ANSWER_QUALITY_NOT_APPLICABLE = "not_applicable"

RAG_RESULT_ROUTE_REVIEW = "review"
RAG_RESULT_ROUTE_NO_ANSWER = "no_answer"

REVIEW_ROUTE_FINAL = "final"
REVIEW_ROUTE_HUMAN_REVIEW = "human_review"
HUMAN_REVIEW_STATUS_REQUIRED = "requires_manual_review"


# 定义流程数据
class AgentState(TypedDict):
    # 输入：Langgraph流程启动时需要代入
    question: str
    top_k: int
    department: str | None
    storage_dir: Path

    # 内部业务域分类，用于调试和后续多 Agent 拆分
    question_type: str
    route: str
    domain_type: str

    # 输出：节点执行后写回State
    answer: str
    sources: list[dict[str, Any]]
    retrieval: dict[str, Any]

    # 工具调用结果：保存结构化数据
    tool_result: dict[str, Any]

    # 内部review状态 ：后续用于判断是否需要人工确认
    answer_review: dict[str, Any]


# 定义节点处理逻辑
def classify_question_node(state: AgentState) -> dict:
    return classify_question(state["question"])


def rag_answer_node(state: AgentState) -> dict:
    result = chat_with_qdrant_retrieval(
        question=state["question"],
        similarity_top_k=state["top_k"],
        department=state["department"],
        domain_type=state["domain_type"],
    )

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "retrieval": result["retrieval"],
    }


def review_answer_node(state: AgentState) -> dict:
    sources = state["sources"]
    retrieval = state["retrieval"]

    has_sources = len(sources) > 0
    used_count = retrieval.get("used_count", 0)

    is_grounded = has_sources and used_count > 0

    answer_quality = ANSWER_QUALITY_GROUNDED if is_grounded else ANSWER_QUALITY_WEAK

    return {
        "answer_review": {
            "answer_quality": answer_quality,
            "needs_review": not is_grounded,
            "has_sources": has_sources,
            "used_count": used_count,
        }
    }


def route_by_rag_result(state: AgentState) -> str:
    # used_count = 0 表示没有证据生成答案，不属于回答质量问题。
    if state["retrieval"].get("used_count", 0) == 0:
        return RAG_RESULT_ROUTE_NO_ANSWER

    return RAG_RESULT_ROUTE_REVIEW


def no_answer_result_node(state: AgentState) -> dict:
    return {
        "answer_review": {
            "answer_quality": ANSWER_QUALITY_NOT_APPLICABLE,
            "needs_review": False,
            "has_sources": False,
            "used_count": 0,
        }
    }


def tool_node(state: AgentState) -> dict:

    work_order_id = extract_work_order_id(state["question"])

    if work_order_id is None:
        tool_result = {
            "tool_name": "query_work_order_status",
            "work_order_id": None,
            "found": False,
            "reason": TOOL_REASON_MISSING_WORK_ORDER_ID,
            "source": "question",
        }
        answer = "请提供工单号，例如 WO-20260727-001。"
    else:
        tool_result = query_work_order_status(work_order_id)

        if not tool_result["found"]:
            # answer 面向用户；tool_result保留结构化查询结果。
            answer = f"未找到工单{tool_result['work_order_id']}。"
        else:
            answer = (
                f"工单 {tool_result['work_order_id']} 当前状态为 {tool_result['status']}。"
                f"产线：{tool_result['line']}，产品：{tool_result['product']}，"
                f"计划数量：{tool_result['planned_quantity']}，"
                f"已完成数量：{tool_result['completed_quantity']}。"
            )

    return {
        "answer": answer,
        "sources": [],
        "retrieval": {
            "top_k": state["top_k"],
            "min_score": MIN_RETRIEVAL_SCORE,
            "retrieved_count": 0,
            "used_count": 0,
        },
        "tool_result": tool_result,
    }


def route_by_answer_review(state: AgentState) -> str:
    answer_review = state["answer_review"]

    if answer_review.get("needs_review"):
        return REVIEW_ROUTE_HUMAN_REVIEW

    return REVIEW_ROUTE_FINAL


def human_review_stub_node(state: AgentState) -> dict:
    answer_review = state["answer_review"]

    return {
        "answer_review": {
            **answer_review,
            "human_review_required": True,
            "human_review_status": HUMAN_REVIEW_STATUS_REQUIRED,
            "human_review_decision": None,
        }
    }


def fallback_node(state: AgentState) -> dict:
    return {
        "answer": NO_ANSWER_MESSAGE,
        "sources": [],
        "retrieval": {
            "top_k": state["top_k"],
            "min_score": MIN_RETRIEVAL_SCORE,
            "retrieved_count": 0,
            "used_count": 0,
        },
    }


def route_by_question_type(state: AgentState) -> str:
    return state["route"]


def build_graph():
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("classify_question", classify_question_node)
    graph_builder.add_node("rag_answer", rag_answer_node)
    graph_builder.add_node("tool_node", tool_node)
    graph_builder.add_node("fallback", fallback_node)
    graph_builder.add_node("no_answer_result", no_answer_result_node)
    graph_builder.add_node("review_answer", review_answer_node)
    graph_builder.add_node("human_review_stub", human_review_stub_node)

    graph_builder.add_edge(START, "classify_question")
    graph_builder.add_conditional_edges(
        "classify_question",
        route_by_question_type,
        {
            ROUTE_RAG_ANSWER: "rag_answer",
            ROUTE_TOOL_NODE: "tool_node",
            ROUTE_FALLBACK: "fallback",
        },
    )
    graph_builder.add_edge("tool_node", END)
    graph_builder.add_conditional_edges(
        "rag_answer",
        route_by_rag_result,
        {
            RAG_RESULT_ROUTE_REVIEW: "review_answer",
            RAG_RESULT_ROUTE_NO_ANSWER: "no_answer_result",
        },
    )
    graph_builder.add_edge("no_answer_result", END)
    graph_builder.add_conditional_edges(
        "review_answer",
        route_by_answer_review,
        {
            REVIEW_ROUTE_FINAL: END,
            REVIEW_ROUTE_HUMAN_REVIEW: "human_review_stub",
        },
    )
    graph_builder.add_edge("human_review_stub", END)
    graph_builder.add_edge("fallback", END)

    return graph_builder.compile()


def run_agent(
    question: str,
    storage_dir: Path,
    top_k: int,
    department: str | None = None,
) -> AgentState:
    graph = build_graph()

    initial_state: AgentState = {
        "question": question,
        "top_k": top_k,
        "department": department,
        "storage_dir": storage_dir,
        "question_type": UNKNOWN,
        "route": "",
        "domain_type": UNKNOWN_DOMAIN,
        "answer": "",
        "sources": [],
        "retrieval": {},
        "tool_result": {},
        "answer_review": {},
    }

    return graph.invoke(initial_state)
