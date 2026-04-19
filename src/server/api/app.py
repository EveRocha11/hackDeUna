"""Minimal FastAPI app for the DeUna backend MVP."""

from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import HTTPException

from server.assistant.models import AssistantQueryRequest, AssistantQueryResponse
from server.assistant.service import execute_assistant_query, load_runtime_config

load_dotenv()
app = FastAPI(title="DeUna Assistant Backend", version="0.1.0")
runtime_config = load_runtime_config()


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health status.

    Args:
        None.

    Returns:
        dict[str, str]: Health payload.

    Raises:
        None.
    """
    return {"status": "ok"}


@app.post("/assistant/query", response_model=AssistantQueryResponse)
def assistant_query(payload: AssistantQueryRequest) -> AssistantQueryResponse:
    """Resolve one assistant query into a deterministic analytical answer.

    Args:
        payload: User query payload.

    Returns:
        AssistantQueryResponse: Query result payload.

    Raises:
        HTTPException: If query processing fails unexpectedly.
    """
    try:
        return execute_assistant_query(payload, runtime_config)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"assistant_query_failed: {exc}") from exc
