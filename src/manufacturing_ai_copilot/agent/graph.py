from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from manufacturing_ai_copilot.core.config import MIN_RETRIEVAL_SCORE, NO_ANSWER_MESSAGE
from manufacturing_ai_copilot.rag.query_engine import chat_with_retrieval
from manufacturing_ai_copilot.agent.classifier import (
    ROUTE_RAG_ANSWER,
    ROUTE_FALLBACK,
    UNKNOWN,
    UNKNOWN_DOMAIN,
    classify_question,
)


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

    # 内部review状态 ：后续用于判断是否需要人工确认
    answer_review: dict[str, Any]


# 定义节点处理逻辑
def classify_question_node(state: AgentState) -> dict:
    return classify_question(state["question"])


def rag_answer_node(state: AgentState) -> dict:
    result = chat_with_retrieval(
        storage_dir=state["storage_dir"],
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

    return {
        "answer_review": {
            "has_sources": has_sources,
            "used_count": used_count,
            "passed": has_sources and used_count > 0,
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
    graph_builder.add_node("fallback", fallback_node)
    graph_builder.add_node("review_answer", review_answer_node)

    graph_builder.add_edge(START, "classify_question")
    graph_builder.add_conditional_edges(
        "classify_question",
        route_by_question_type,
        {
            ROUTE_RAG_ANSWER: "rag_answer",
            ROUTE_FALLBACK: "fallback",
        },
    )

    graph_builder.add_edge("rag_answer", "review_answer")
    graph_builder.add_edge("review_answer", END)
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
        "answer_review": {},
    }

    return graph.invoke(initial_state)
