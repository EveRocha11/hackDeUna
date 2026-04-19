"""LangGraph compiled conversational agent graph."""

from __future__ import annotations

import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from server.langgraph_agent.prompts import DEUNA_AGENT_SYSTEM_PROMPT
from server.langgraph_agent.tools import (
    assistant_capabilities_tool,
    run_analytics_query_tool,
    run_sql_analytics_tool,
)


def _build_model() -> ChatOpenAI:
    """Build chat model instance for conversational agent runtime.

    Args:
        None.

    Returns:
        ChatOpenAI: Initialized model client.

    Raises:
        ValueError: If numeric environment values are invalid.
    """
    return ChatOpenAI(
        model=os.getenv("AGENT_MODEL", "gpt-4.1-mini"),
        temperature=float(os.getenv("AGENT_TEMPERATURE", "0.0")),
        timeout=float(os.getenv("AGENT_TIMEOUT_SECONDS", "12.0")),
    )


graph = create_agent(
    model=_build_model(),
    tools=[assistant_capabilities_tool, run_analytics_query_tool, run_sql_analytics_tool],
    system_prompt=DEUNA_AGENT_SYSTEM_PROMPT,
    name="deuna_conversational_agent",
    checkpointer=MemorySaver(),
)
