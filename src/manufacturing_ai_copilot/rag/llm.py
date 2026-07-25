import os

from openai import OpenAI
from manufacturing_ai_copilot.core.config import DASHSCOPE_BASE_URL, LLM_MODEL

DOMAIN_ANSWER_INSTRUCTIONS = {
    "equipment_sop": "请按设备SOP风格回答，优先给出处理步骤、注意事项、复发升级条件。",
    "production_order": "请按生产工单处理风格回答，说明工单状态含义、处理步骤、需要确认的数据。",
    "quality_issue": "请按质量问题分析风格回答，优先说明现象确认、临时措施、根因分析、永久措施。",
    "it_support": "请按IT支持风格回答，优先给出操作步骤、权限/账号注意事项、安全提醒。",
    "general_knowledge": "请基于知识库内容直接回答，保持简洁，并避免编造未提供的信息。",
}


def get_answer_instruction(domain_type: str) -> str:

    return DOMAIN_ANSWER_INSTRUCTIONS.get(
        domain_type,
        DOMAIN_ANSWER_INSTRUCTIONS["general_knowledge"],
    )


def _build_context(matches: list[dict]) -> str:
    context_parts = []

    for index, match in enumerate(matches, start=1):
        title = match.get("title") or "未知文档"
        document = match.get("document") or "unknown"
        content = match.get("content") or ""

        context_parts.append(
            f"[{index}] title: {title}\n"
            f"document: {document}\n"
            f"content:\n{content}"
        )

    return "\n\n".join(context_parts)


def generate_answer_with_qwen(
    question: str,
    matches: list[dict],
    domain_type: str = "general_knowledge",
) -> str:
    if not matches:
        raise ValueError(
            "generate_answer_with_qwen requires at least one retrieved match."
        )

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set.")

    client = OpenAI(
        api_key=api_key,
        base_url=DASHSCOPE_BASE_URL,
    )

    context = _build_context(matches)

    answer_instruction = get_answer_instruction(domain_type)

    system_prompt = (
        "你是制造企业知识库助手。"
        "只能基于给定的 context 回答问题。"
        "如果 context 中没有答案，明确说明未找到依据。"
        "回答要简洁、可执行，并保留必要的操作步骤。"
        f"\n\n回答策略：{answer_instruction}"
    )

    user_prompt = (
        f"问题：{question}\n\n" f"context：\n{context}\n\n" "请基于 context 回答问题。"
    )

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return completion.choices[0].message.content or ""
