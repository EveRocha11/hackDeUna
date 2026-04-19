"""Intent classification helpers with catalog-constrained LLM output."""

from __future__ import annotations

from pathlib import Path

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import yaml


class IntentClassification(BaseModel):
    """Structured result returned by intent classifier.

    Args:
        intent_id: Predicted intent id.
        confidence: Confidence score in range 0 to 1.
        clarification_question_es: Optional follow-up clarification in Spanish.

    Returns:
        IntentClassification: Parsed and validated result.

    Raises:
        ValueError: If validation fails.
    """

    intent_id: str = Field(description="Intent id predicted for the question")
    confidence: float = Field(description="Confidence score between 0 and 1")
    clarification_question_es: str | None = Field(
        default=None,
        description="Optional clarification in Spanish when confidence is low",
    )


def load_allowed_intents(path: Path) -> tuple[str, ...]:
    """Load allowed intent ids from semantic catalog YAML.

    Args:
        path: Path to intents catalog file.

    Returns:
        tuple[str, ...]: Sorted unique intent ids.

    Raises:
        FileNotFoundError: If catalog path does not exist.
        ValueError: If YAML content is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Intents catalog not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Intents catalog must be a mapping")

    intents = raw.get("intents")
    if not isinstance(intents, list):
        raise ValueError("Intents catalog must include an 'intents' list")

    allowed: set[str] = set()
    for item in intents:
        if not isinstance(item, dict):
            raise ValueError("Each intent entry must be a mapping")
        intent_id = str(item.get("id", "")).strip()
        if not intent_id:
            raise ValueError("Each intent entry must include non-empty id")
        allowed.add(intent_id)

    return tuple(sorted(allowed))


def classify_intent_with_llm(
    question_es: str,
    allowed_intents: tuple[str, ...],
    model: str,
    temperature: float,
    timeout_seconds: float,
) -> IntentClassification | None:
    """Classify user question into one allowed intent using a small LLM.

    Args:
        question_es: User question in Spanish.
        allowed_intents: Allowed intent ids from semantic catalog.
        model: LLM model name.
        temperature: LLM temperature.
        timeout_seconds: LLM call timeout in seconds.

    Returns:
        IntentClassification | None: Classification result or None on failure.

    Raises:
        None.
    """
    if not allowed_intents:
        return None

    client = ChatOpenAI(
        model=model,
        temperature=temperature,
        timeout=timeout_seconds,
    )

    allowed_str = ", ".join(allowed_intents)
    prompt = (
        "Clasifica una pregunta de un microcomercio en exactamente uno de los intents permitidos.\n"
        "No inventes intent IDs.\n"
        "Si hay duda, elige el intent más cercano y baja confidence.\n"
        "Devuelve JSON válido con las claves: intent_id, confidence, clarification_question_es.\n"
        f"Intents permitidos: {allowed_str}\n"
        f"Pregunta: {question_es}"
    )

    try:
        structured = client.with_structured_output(IntentClassification)
        result = structured.invoke(prompt)
        if isinstance(result, dict):
            parsed = IntentClassification.model_validate(result)
        else:
            parsed = result

        if parsed.intent_id not in allowed_intents:
            return None

        confidence = max(0.0, min(1.0, float(parsed.confidence)))
        return IntentClassification(
            intent_id=parsed.intent_id,
            confidence=confidence,
            clarification_question_es=parsed.clarification_question_es,
        )
    except Exception:
        return None
