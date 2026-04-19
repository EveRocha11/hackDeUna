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


class AgentQueryRequest(BaseModel):
    """Input payload for one LangGraph conversational agent turn.

    Args:
        question_es: User question in Spanish.
        thread_id: Optional thread id to keep conversational state.

    Returns:
        AgentQueryRequest: Validated request payload.

    Raises:
        ValueError: If payload validation fails.
    """

    question_es: str = Field(min_length=1, description="Pregunta del usuario")
    thread_id: str | None = Field(
        default=None,
        description="Optional thread identifier for multi-turn memory",
    )


class AgentQueryResponse(BaseModel):
    """Output payload returned by LangGraph conversational agent endpoint.

    Args:
        answer_es: Final assistant answer in Spanish.
        tool_call_names: Ordered tool call names emitted by trajectory.
        message_count: Number of messages in the returned trajectory.

    Returns:
        AgentQueryResponse: Serialized response payload.

    Raises:
        ValueError: If output validation fails.
    """

    answer_es: str
    tool_call_names: list[str]
    message_count: int
