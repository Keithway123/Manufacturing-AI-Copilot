from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from manufacturing_ai_copilot.core.config import MIN_RETRIEVAL_SCORE, NO_ANSWER_MESSAGE
from manufacturing_ai_copilot.rag.query_engine import chat_with_retrieval

KNOWLEDGE_QA = "knowledge_qa"
UNKNOWN = "unknown"
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


# 定义流程数据
class AgentState(TypedDict):
    # 输入：Langgraph流程启动时需要代入
    question: str
    top_k: int
    department: str | None
    storage_dir: Path

    question_type: str

    # 输出：节点执行后写回State
    answer: str
    sources: list[dict[str, Any]]
    retrieval: dict[str, Any]


# 定义节点处理逻辑
def classify_question_node(state: AgentState) -> dict:
    question = state["question"]

    question_type = UNKNOWN
    for keyword in KNOWLEDGE_QA_KEYWORDS:
        if keyword.lower() in question.lower():
            question_type = KNOWLEDGE_QA
            break

    return {
        "question_type": question_type,
    }


def rag_answer_node(state: AgentState) -> dict:
    result = chat_with_retrieval(
        storage_dir=state["storage_dir"],
        question=state["question"],
        similarity_top_k=state["top_k"],
        department=state["department"],
    )

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "retrieval": result["retrieval"],
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
    if state["question_type"] == KNOWLEDGE_QA:
        return "rag_answer"
    return "fallback"


def build_graph():
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("classify_question", classify_question_node)
    graph_builder.add_node("rag_answer", rag_answer_node)
    graph_builder.add_node("fallback", fallback_node)

    graph_builder.add_edge(START, "classify_question")
    graph_builder.add_conditional_edges(
        "classify_question",
        route_by_question_type,
        {
            "rag_answer": "rag_answer",
            "fallback": "fallback",
        },
    )

    graph_builder.add_edge("rag_answer", END)
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
        "answer": "",
        "sources": [],
        "retrieval": {},
    }

    return graph.invoke(initial_state)
