"""Answer composition helpers for the assistant agent layer."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_openai import ChatOpenAI


def generate_answer_from_facts(
    *,
    question_es: str,
    draft_answer_es: str,
    facts_payload: dict[str, object],
    evidence_payload: dict[str, object],
    proactive_flags: list[str],
    model: str,
    temperature: float,
    timeout_seconds: float,
    system_prompt_path: Path,
) -> str | None:
    """Generate a final Spanish answer grounded on deterministic facts.

    Args:
        question_es: Original user question.
        draft_answer_es: Deterministic draft answer.
        facts_payload: Structured facts returned by SQL.
        evidence_payload: Query/evidence metadata.
        proactive_flags: Optional proactive signals.
        model: LLM model name.
        temperature: LLM temperature.
        timeout_seconds: Request timeout in seconds.
        system_prompt_path: Path to system prompt file.

    Returns:
        str | None: Final answer, or None when generation fails.

    Raises:
        None.
    """
    if not system_prompt_path.exists():
        return None

    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        return None

    user_payload = {
        "question_es": question_es,
        "draft_answer_es": draft_answer_es,
        "facts_payload": facts_payload,
        "evidence_payload": evidence_payload,
        "proactive_flags": proactive_flags,
    }

    user_prompt = (
        "Usa exclusivamente el payload para responder. "
        "Si draft_answer_es ya es correcto, mejóralo solo en claridad y concisión.\n"
        f"{json.dumps(user_payload, ensure_ascii=False, indent=2)}"
    )

    try:
        client = ChatOpenAI(
            model=model,
            temperature=temperature,
            timeout=timeout_seconds,
        )
        response = client.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        text = str(response.content).strip()
        return text or None
    except Exception:
        return None
