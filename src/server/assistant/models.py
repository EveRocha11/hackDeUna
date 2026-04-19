"""Pydantic models for assistant query endpoint payloads."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class AssistantQueryRequest(BaseModel):
    """Input payload for one assistant query.

    Args:
        question_es: User question in Spanish.
        merchant_id: Optional merchant override.
        as_of_date: Optional reference date for period calculations.

    Returns:
        AssistantQueryRequest: Validated request payload.

    Raises:
        ValueError: If required fields fail validation.
    """

    question_es: str = Field(min_length=3, description="Pregunta del comercio en español")
    merchant_id: str | None = Field(
        default=None,
        description="Merchant override. If missing, DEUNA_MERCHANT_ID or M001 is used.",
    )
    as_of_date: date | None = Field(
        default=None,
        description="Optional reference date to compute week/month windows.",
    )


class AssistantQueryResponse(BaseModel):
    """Output payload returned by the assistant endpoint.

    Args:
        status: Query status (ok, clarification, unsupported).
        merchant_id: Merchant associated with the answer.
        intent_id: Resolved intent id.
        answer_es: Final answer in Spanish.
        facts_payload: Structured facts used by the answer.
        evidence_payload: Query metadata for observability.
        intent_source: Routing source used for detected intent.
        intent_confidence: Confidence from classifier when available.
        chart_allowed: Whether chart rendering is allowed.
        proactive_flags: Optional list of proactive hints.
        clarification_question_es: Optional follow-up question.

    Returns:
        AssistantQueryResponse: Serialized response payload.

    Raises:
        ValueError: If output validation fails.
    """

    status: str
    merchant_id: str
    intent_id: str
    answer_es: str
    facts_payload: dict[str, Any]
    evidence_payload: dict[str, Any]
    intent_source: str
    intent_confidence: float | None = None
    chart_allowed: bool
    proactive_flags: list[str]
    clarification_question_es: str | None = None
