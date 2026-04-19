"""Minimal FastAPI app for the DeUna backend MVP."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, BaseMessage

from server.assistant.models import (
    AgentQueryRequest,
    AgentQueryResponse,
    AssistantQueryRequest,
    AssistantQueryResponse,
    FrontendAgentQueryRequest,
    FrontendAgentQueryResponse,
)
from server.assistant.service import execute_assistant_query, load_runtime_config
from server.langgraph_agent.graph import graph

load_dotenv()
app = FastAPI(title="DeUna Assistant Backend", version="0.1.0")
runtime_config = load_runtime_config()

cors_origins_raw = os.getenv("API_CORS_ALLOW_ORIGINS", "*").strip()
cors_origins = [origin.strip() for origin in cors_origins_raw.split(",") if origin.strip()]
if not cors_origins:
    cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_agent_query(question_es: str, thread_id: str | None) -> AgentQueryResponse:
    """Execute one agent query and normalize the response payload.

    Args:
        question_es: User question in Spanish.
        thread_id: Optional thread id for conversational continuity.

    Returns:
        AgentQueryResponse: Normalized runtime result.

    Raises:
        HTTPException: If the runtime invocation fails.
    """
    effective_thread_id = thread_id or "default-thread"
    try:
        run_output = graph.invoke(
            {"messages": [{"role": "user", "content": question_es}]},
            config={"configurable": {"thread_id": effective_thread_id}},
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"agent_query_failed: {exc}") from exc

    messages_raw = run_output.get("messages")
    if not isinstance(messages_raw, list):
        raise HTTPException(status_code=500, detail="agent_query_failed: missing messages")

    messages: list[BaseMessage] = messages_raw
    answer_es = ""
    tool_call_names: list[str] = []

    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls:
                name = call.get("name")
                if isinstance(name, str) and name:
                    tool_call_names.append(name)
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            answer_es = msg.content.strip()
            break

    return AgentQueryResponse(
        answer_es=answer_es,
        tool_call_names=tool_call_names,
        message_count=len(messages),
    )


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


@app.post("/assistant/agent-query", response_model=AgentQueryResponse)
def assistant_agent_query(payload: AgentQueryRequest) -> AgentQueryResponse:
    """Resolve one query using LangGraph create_agent runtime.

    Args:
        payload: User query payload.

    Returns:
        AgentQueryResponse: Conversational agent result payload.

    Raises:
        HTTPException: If runtime invocation fails.
    """
    return _run_agent_query(
        question_es=payload.question_es,
        thread_id=payload.thread_id,
    )


@app.post("/api/v1/agent/query", response_model=FrontendAgentQueryResponse)
def frontend_agent_query(payload: FrontendAgentQueryRequest) -> FrontendAgentQueryResponse:
    """Minimal frontend-facing conversational endpoint.

    Args:
        payload: Frontend-friendly payload with plain `question`.

    Returns:
        FrontendAgentQueryResponse: Answer plus tool trace names.

    Raises:
        HTTPException: If runtime invocation fails.
    """
    agent_result = _run_agent_query(
        question_es=payload.question,
        thread_id=payload.thread_id,
    )
    return FrontendAgentQueryResponse(
        answer=agent_result.answer_es,
        tools=agent_result.tool_call_names,
    )
