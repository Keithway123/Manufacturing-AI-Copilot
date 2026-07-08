import os

from openai import OpenAI
from manufacturing_ai_copilot.core.config import DASHSCOPE_BASE_URL, LLM_MODEL


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


def generate_answer_with_qwen(question: str, matches: list[dict]) -> str:
    if not matches:
        return "未在当前知识库中找到相关内容。"

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set.")

    client = OpenAI(
        api_key=api_key,
        base_url=DASHSCOPE_BASE_URL,
    )

    context = _build_context(matches)

    system_prompt = (
        "你是制造企业知识库助手。"
        "只能基于给定的 context 回答问题。"
        "如果 context 中没有答案，明确说明未找到依据。"
        "回答要简洁、可执行，并保留必要的操作步骤。"
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
