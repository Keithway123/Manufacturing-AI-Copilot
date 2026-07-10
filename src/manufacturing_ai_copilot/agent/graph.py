from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from manufacturing_ai_copilot.rag.query_engine import chat_with_retrieval


# 定义流程数据
class AgentState(TypedDict):
    # 输入：Langgraph流程启动时需要代入
    question: str
    top_k: int
    department: str | None
    storage_dir: Path

    # 输出：节点执行后写回State
    answer: str
    sources: list[dict[str, Any]]
    retrieval: dict[str, Any]


# 定义节点处理逻辑
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


def build_graph():
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("rag_answer", rag_answer_node)
    graph_builder.add_edge(START, "rag_answer")
    graph_builder.add_edge("rag_answer", END)

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
        "answer": "",
        "sources": [],
        "retrieval": {},
    }

    return graph.invoke(initial_state)
