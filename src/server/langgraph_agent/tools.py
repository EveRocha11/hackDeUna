"""Tool definitions for the LangGraph conversational agent."""

from __future__ import annotations

import json
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
import unicodedata

import duckdb
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import yaml

from server.assistant.models import AssistantQueryRequest
from server.assistant.service import execute_assistant_query, load_runtime_config
from server.assistant.text2sql import TextToSQLConfig, execute_guarded_sql, run_text_to_sql_fallback


_ASSISTANT_RUNTIME_CONFIG = load_runtime_config()


class QueryPlanDraft(BaseModel):
    """Structured output for registry query planning.

    Args:
        query_key: Selected registry query key or __NO_MATCH__.
        confidence: Confidence score from 0.0 to 1.0.
        reason_es: Short explanation in Spanish.

    Returns:
        QueryPlanDraft: Parsed planner result.

    Raises:
        ValueError: If fields violate schema constraints.
    """

    query_key: str = Field(description="Exact query_key from candidates, or __NO_MATCH__")
    confidence: float = Field(ge=0.0, le=1.0)
    reason_es: str = Field(description="Breve motivo de seleccion")


def _duckdb_path() -> Path:
    """Resolve analytics DuckDB path from environment.

    Args:
        None.

    Returns:
        Path: Effective DuckDB file path.

    Raises:
        None.
    """
    return Path(os.getenv("DATA_DUCKDB_PATH", "data/duckdb/analytics.duckdb"))


def _default_merchant_id() -> str:
    """Resolve default merchant id from environment.

    Args:
        None.

    Returns:
        str: Merchant id.

    Raises:
        None.
    """
    return (os.getenv("DEUNA_MERCHANT_ID") or "M001").strip()


def _query_registry_path() -> Path:
    """Resolve query registry path from environment.

    Args:
        None.

    Returns:
        Path: Effective query registry path.

    Raises:
        None.
    """
    return Path(os.getenv("QUERY_REGISTRY_PATH", "src/agent/semantics/query_registry.yaml"))


def _load_query_registry() -> dict[str, Any]:
    """Load registry payload with query capability definitions.

    Args:
        None.

    Returns:
        dict[str, Any]: Parsed registry payload.

    Raises:
        FileNotFoundError: If registry file is missing.
        ValueError: If registry payload has invalid shape.
    """
    path = _query_registry_path()
    if not path.exists():
        raise FileNotFoundError(f"Query registry not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Query registry must be a mapping")
    queries = raw.get("queries")
    if not isinstance(queries, list):
        raise ValueError("Query registry must include a 'queries' list")
    return raw


def _enabled_registry_queries(registry: dict[str, Any]) -> list[dict[str, Any]]:
    """Return enabled query entries from registry.

    Args:
        registry: Parsed query registry payload.

    Returns:
        list[dict[str, Any]]: Enabled entries only.

    Raises:
        None.
    """
    queries = registry.get("queries", [])
    if not isinstance(queries, list):
        return []
    enabled: list[dict[str, Any]] = []
    for entry in queries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("status", "")).strip() == "enabled":
            enabled.append(entry)
    return enabled


def _planner_model() -> ChatOpenAI:
    """Build planner model client for query routing decisions.

    Args:
        None.

    Returns:
        ChatOpenAI: Planner model client.

    Raises:
        ValueError: If numeric environment values are invalid.
    """
    return ChatOpenAI(
        model=os.getenv("AGENT_PLANNER_MODEL", os.getenv("AGENT_MODEL", "gpt-4.1-mini")),
        temperature=float(os.getenv("AGENT_PLANNER_TEMPERATURE", "0.0")),
        timeout=float(os.getenv("AGENT_PLANNER_TIMEOUT_SECONDS", "6.0")),
    )


def _dataset_as_of_date() -> date:
    """Resolve dataset reference date from environment.

    Args:
        None.

    Returns:
        date: Reference date used by default slot inference.

    Raises:
        ValueError: If date format is invalid.
    """
    return date.fromisoformat(os.getenv("DATASET_END_DATE", "2026-04-18"))


def _normalize_text(value: str) -> str:
    """Normalize text for robust keyword matching.

    Args:
        value: Raw text.

    Returns:
        str: Lowercased accent-insensitive text.

    Raises:
        None.
    """
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return ascii_text.lower().strip()


def _json_safe(value: Any) -> Any:
    """Convert values into JSON-serializable structures.

    Args:
        value: Raw object.

    Returns:
        Any: JSON-safe value.

    Raises:
        None.
    """
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _find_registry_entry_by_query_key(
    registry: dict[str, Any],
    query_key: str,
) -> dict[str, Any] | None:
    """Find enabled registry entry by exact query key.

    Args:
        registry: Registry payload.
        query_key: Query key to resolve.

    Returns:
        dict[str, Any] | None: Matching enabled entry.

    Raises:
        None.
    """
    normalized_key = query_key.strip()
    for entry in _enabled_registry_queries(registry):
        if str(entry.get("query_key", "")).strip() == normalized_key:
            return entry
    return None


def _infer_time_slots(intent_id: str, question_es: str, slots: dict[str, Any]) -> dict[str, Any]:
    """Infer common period slots from natural language when missing.

    Args:
        intent_id: Target intent identifier.
        question_es: User question.
        slots: Input slots payload.

    Returns:
        dict[str, Any]: Slots payload with inferred dates.

    Raises:
        None.
    """
    as_of = _dataset_as_of_date()
    question = _normalize_text(question_es)
    inferred: dict[str, Any]

    if intent_id in {"customer_new_vs_returning", "top_customers"}:
        month_start = as_of.replace(day=1)
        inferred = {
            "start_date": month_start.isoformat(),
            "end_date": as_of.isoformat(),
        }
    elif "ayer" in question:
        d = as_of - timedelta(days=1)
        inferred = {
            "start_date": d.isoformat(),
            "end_date": d.isoformat(),
        }
    elif "mes" in question:
        month_start = as_of.replace(day=1)
        inferred = {
            "start_date": month_start.isoformat(),
            "end_date": as_of.isoformat(),
        }
    else:
        week_start = as_of - timedelta(days=as_of.weekday())
        week_end = week_start + timedelta(days=6)
        inferred = {
            "start_date": week_start.isoformat(),
            "end_date": week_end.isoformat(),
        }

    if intent_id == "income_vs_previous":
        week_start = as_of - timedelta(days=as_of.weekday())
        week_end = week_start + timedelta(days=6)
        inferred = {
            "current_start": week_start.isoformat(),
            "current_end": week_end.isoformat(),
            "previous_start": (week_start - timedelta(days=7)).isoformat(),
            "previous_end": (week_end - timedelta(days=7)).isoformat(),
        }
    elif intent_id == "inactive_customers":
        inferred = {
            "end_date": as_of.isoformat(),
            "recent_start": (as_of - timedelta(days=29)).isoformat(),
        }

    merged = dict(inferred)
    merged.update(slots)
    return merged


def _plan_query_entry_with_llm(
    *,
    registry: dict[str, Any],
    question_es: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Plan best registry query entry for a question using LLM.

    Args:
        registry: Query registry payload.
        question_es: User question in Spanish.

    Returns:
        tuple[dict[str, Any] | None, dict[str, Any]]: Selected entry and planner metadata.

    Raises:
        None.
    """
    candidates = _enabled_registry_queries(registry)
    if not candidates:
        return None, {"error": "no_enabled_queries"}
    if not os.getenv("OPENAI_API_KEY"):
        return None, {"error": "missing_openai_api_key"}

    candidate_view = [
        {
            "query_key": str(entry.get("query_key", "")).strip(),
            "intent_id": str(entry.get("intent_id", "")).strip(),
            "description_es": str(entry.get("description_es", "")).strip(),
            "required_slots": entry.get("required_slots", []),
            "optional_slots": entry.get("optional_slots", []),
            "tags": entry.get("capability_tags", []),
        }
        for entry in candidates
    ]
    query_keys = [item["query_key"] for item in candidate_view]
    unsupported_topics = registry.get("unsupported_topics", [])

    prompt = (
        "Selecciona el query_key mas adecuado para responder la pregunta del usuario.\n"
        "Reglas:\n"
        "- Debes devolver un query_key EXACTO de la lista.\n"
        "- Si no hay match razonable, devuelve __NO_MATCH__.\n"
        "- Si la pregunta trata un tema fuera de alcance, devuelve __NO_MATCH__.\n"
        "- No inventes valores.\n\n"
        f"Pregunta: {question_es}\n"
        f"Temas fuera de alcance: {json.dumps(unsupported_topics, ensure_ascii=False)}\n"
        f"Candidatos JSON: {json.dumps(candidate_view, ensure_ascii=False)}"
    )

    try:
        planner = _planner_model().with_structured_output(QueryPlanDraft)
        draft = planner.invoke(prompt)
        if isinstance(draft, dict):
            query_key = str(draft.get("query_key", "")).strip()
            confidence = float(draft.get("confidence", 0.0))
            reason_es = str(draft.get("reason_es", "")).strip()
        else:
            query_key = draft.query_key.strip()
            confidence = float(draft.confidence)
            reason_es = draft.reason_es.strip()
    except Exception as exc:
        return None, {"error": "planner_failed", "detail": str(exc)}

    planner_info = {
        "query_key": query_key,
        "confidence": confidence,
        "reason_es": reason_es,
        "candidate_count": len(candidate_view),
    }

    if query_key == "__NO_MATCH__":
        return None, planner_info
    if query_key not in query_keys:
        planner_info["error"] = "planner_query_key_not_in_registry"
        return None, planner_info

    min_confidence = float(os.getenv("AGENT_PLANNER_MIN_CONFIDENCE", "0.55"))
    if confidence < min_confidence:
        planner_info["error"] = "planner_low_confidence"
        planner_info["min_confidence"] = min_confidence
        return None, planner_info

    selected = _find_registry_entry_by_query_key(registry, query_key)
    if selected is None:
        planner_info["error"] = "planner_query_not_enabled"
        return None, planner_info
    return selected, planner_info


def _sql_literal(value: Any) -> str:
    """Convert scalar value into SQL literal string.

    Args:
        value: Raw scalar value.

    Returns:
        str: SQL literal-safe representation.

    Raises:
        ValueError: If value type is unsupported.
    """
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return escaped
    raise ValueError(f"Unsupported slot value type: {type(value)!r}")


def _render_registry_sql(entry: dict[str, Any], merchant_id: str, slots: dict[str, Any]) -> str:
    """Render SQL template with merchant and slot literals.

    Args:
        entry: Registry query entry.
        merchant_id: Effective merchant id.
        slots: Parsed slots payload.

    Returns:
        str: Rendered SQL statement.

    Raises:
        ValueError: If template or required fields are invalid.
    """
    template = entry.get("sql_template")
    if not isinstance(template, str) or not template.strip():
        raise ValueError("Registry entry missing sql_template")

    required = entry.get("required_slots", [])
    if not isinstance(required, list):
        raise ValueError("required_slots must be a list")

    bindings: dict[str, str] = {"merchant_id": _sql_literal(merchant_id)}
    for key in required:
        if key not in slots:
            raise ValueError(f"Missing required slot: {key}")
        bindings[str(key)] = _sql_literal(slots[str(key)])
    for key, value in slots.items():
        if str(key) not in bindings:
            bindings[str(key)] = _sql_literal(value)

    return template.format(**bindings)


def _find_registry_entry(
    registry: dict[str, Any],
    intent_id: str,
    question_es: str,
) -> dict[str, Any] | None:
    """Find first enabled query entry for an intent.

    Args:
        registry: Registry payload.
        intent_id: Intent identifier.

    Returns:
        dict[str, Any] | None: Matching enabled entry.

    Raises:
        None.
    """
    queries = registry.get("queries", [])
    if not isinstance(queries, list):
        return None
    candidates: list[dict[str, Any]] = []
    for entry in queries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("intent_id", "")).strip() != intent_id:
            continue
        if str(entry.get("status", "")).strip() == "enabled":
            candidates.append(entry)

    if not candidates:
        return None

    question = _normalize_text(question_es)
    scored: list[tuple[int, dict[str, Any]]] = []
    for entry in candidates:
        keywords = entry.get("match_any_keywords", [])
        if not isinstance(keywords, list):
            keywords = []
        if not keywords:
            scored.append((0, entry))
            continue

        hit_count = 0
        for raw_kw in keywords:
            kw = _normalize_text(str(raw_kw))
            if kw and kw in question:
                hit_count += 1
        scored.append((hit_count, entry))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _build_facts_payload(
    query_key: str, columns: list[str], rows: list[list[Any]]
) -> dict[str, Any]:
    """Build query-key specific facts payload for answer synthesis quality.

    Args:
        query_key: Executed query key.
        columns: Column names.
        rows: Returned row values.

    Returns:
        dict[str, Any]: Structured facts payload.

    Raises:
        None.
    """
    rows_dict = [
        {str(col): _json_safe(row[idx]) for idx, col in enumerate(columns)} for row in rows
    ]

    if query_key == "daily_income_week":
        return {
            "daily_income": [
                {"date": str(item.get("date")), "income": round(float(item.get("income", 0.0)), 2)}
                for item in rows_dict
            ],
            "currency": "USD",
        }
    if query_key == "seller_performance_week":
        return {
            "seller_breakdown": [
                {
                    "seller_id": item.get("seller_id"),
                    "seller_display_name": item.get("seller_display_name"),
                    "role": item.get("role"),
                    "income": round(float(item.get("income", 0.0)), 2),
                    "tx_count": int(item.get("tx_count", 0)),
                }
                for item in rows_dict
            ]
        }
    if query_key == "top_customers_income_month_top3":
        return {
            "top_customers": [
                {
                    "customer_id": item.get("customer_id"),
                    "customer_display_name": item.get("customer_display_name"),
                    "income": round(float(item.get("income", 0.0)), 2),
                }
                for item in rows_dict
            ]
        }
    if query_key == "income_period_total_week" and rows_dict:
        return {
            "total_income": round(float(rows_dict[0].get("total_income", 0.0)), 2),
            "currency": "USD",
        }
    if query_key == "peak_hour_income_week" and rows_dict:
        return {
            "peak_hour": int(rows_dict[0].get("peak_hour", 0)),
            "peak_hour_income": round(float(rows_dict[0].get("peak_hour_income", 0.0)), 2),
            "currency": "USD",
        }
    return {
        "columns": columns,
        "rows": rows_dict,
        "row_count": len(rows_dict),
    }


def _forced_text2sql_config() -> TextToSQLConfig:
    """Build text-to-SQL config with fallback enabled.

    Args:
        None.

    Returns:
        TextToSQLConfig: Fallback configuration.

    Raises:
        None.
    """
    base = _ASSISTANT_RUNTIME_CONFIG.text2sql
    return TextToSQLConfig(
        enabled=True,
        model=base.model,
        temperature=base.temperature,
        timeout_seconds=base.timeout_seconds,
        max_rows=base.max_rows,
    )


def _auto_guarded_sql_fallback(
    *,
    question_es: str,
    merchant_id: str,
) -> dict[str, Any] | None:
    """Attempt automatic guarded text-to-SQL fallback.

    Args:
        question_es: User question.
        merchant_id: Effective merchant id.

    Returns:
        dict[str, Any] | None: Fallback payload when available.

    Raises:
        None.
    """
    with duckdb.connect(str(_duckdb_path()), read_only=True) as conn:
        fallback = run_text_to_sql_fallback(
            conn=conn,
            question_es=question_es,
            merchant_id=merchant_id,
            as_of_date=_dataset_as_of_date(),
            config=_forced_text2sql_config(),
        )
    if fallback is None:
        return None

    rows_safe = _json_safe(fallback.rows)
    return {
        "ok": True,
        "intent_id": "text2sql_fallback",
        "query_key": "text2sql_fallback_auto",
        "merchant_id": merchant_id,
        "resolved_sql": fallback.sql,
        "params": {},
        "summary": fallback.answer_es,
        "columns": fallback.columns,
        "rows": rows_safe,
        "row_count": len(fallback.rows),
        "facts_payload": {
            "columns": fallback.columns,
            "rows": rows_safe,
            "row_count": len(fallback.rows),
        },
        "chart_allowed": False,
        "fallback_used": True,
    }


@tool
def run_analytics_query_tool(
    question_es: str,
    query_key: str | None = None,
    intent_id: str | None = None,
    slots_json: str = "{}",
    merchant_id: str | None = None,
    max_rows: int = 20,
    allow_fallback_sql: bool = True,
) -> str:
    """Run one traceable analytics query resolved from registry by intent.

    Args:
        question_es: User question in Spanish.
        query_key: Optional explicit query key from registry.
        intent_id: Optional semantic intent id (legacy compatibility).
        slots_json: Optional JSON object with slot values.
        merchant_id: Merchant override, defaults to configured merchant.
        max_rows: Maximum number of rows to return.
        allow_fallback_sql: Enable automatic safeguarded SQL fallback when routing fails.

    Returns:
        str: JSON with query provenance, sql, facts payload and rows.

    Raises:
        None.
    """
    effective_merchant = (merchant_id or _default_merchant_id()).strip()
    capped_rows = max(1, min(int(max_rows), 50))
    normalized_intent = "" if intent_id is None else intent_id.strip()
    normalized_query_key = "" if query_key is None else query_key.strip()

    try:
        parsed_slots = json.loads(slots_json) if slots_json.strip() else {}
    except Exception:
        parsed_slots = {}
    if not isinstance(parsed_slots, dict):
        parsed_slots = {}

    try:
        registry = _load_query_registry()
    except Exception as exc:
        return json.dumps(
            {
                "ok": False,
                "error": "query_registry_load_failed",
                "detail": str(exc),
            },
            ensure_ascii=False,
        )

    entry: dict[str, Any] | None = None
    planner_info: dict[str, Any] = {}

    if normalized_query_key:
        entry = _find_registry_entry_by_query_key(registry, normalized_query_key)
        if entry is None:
            planner_info = {
                "error": "query_key_not_enabled",
                "query_key": normalized_query_key,
            }
    elif normalized_intent:
        entry = _find_registry_entry(registry, normalized_intent, question_es)
        if entry is None:
            planner_info = {
                "error": "intent_not_enabled",
                "intent_id": normalized_intent,
            }
    else:
        entry, planner_info = _plan_query_entry_with_llm(
            registry=registry,
            question_es=question_es,
        )

    auto_fallback_enabled = os.getenv("AUTO_GUARDED_SQL_FALLBACK", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if entry is None:
        if allow_fallback_sql and auto_fallback_enabled:
            fallback_payload = _auto_guarded_sql_fallback(
                question_es=question_es,
                merchant_id=effective_merchant,
            )
            if fallback_payload is not None:
                fallback_payload["planner"] = planner_info
                return json.dumps(_json_safe(fallback_payload), ensure_ascii=False)

        return json.dumps(
            {
                "ok": False,
                "error": "routing_failed",
                "intent_id": normalized_intent,
                "query_key": normalized_query_key,
                "planner": planner_info,
                "hint": "Consulta assistant_capabilities_tool para ver capacidades habilitadas.",
            },
            ensure_ascii=False,
        )

    resolved_intent = str(entry.get("intent_id", "")).strip()
    resolved_query_key = str(entry.get("query_key", "")).strip()
    inferred_slots = _infer_time_slots(resolved_intent, question_es, parsed_slots)

    try:
        sql = _render_registry_sql(entry, effective_merchant, inferred_slots)
    except Exception as exc:
        return json.dumps(
            {
                "ok": False,
                "error": "slot_validation_failed",
                "intent_id": resolved_intent,
                "query_key": resolved_query_key,
                "planner": planner_info,
                "detail": str(exc),
            },
            ensure_ascii=False,
        )

    with duckdb.connect(str(_duckdb_path()), read_only=True) as conn:
        result = execute_guarded_sql(
            conn=conn,
            sql=sql,
            merchant_id=effective_merchant,
            max_rows=capped_rows,
        )

    if result is None:
        if allow_fallback_sql and auto_fallback_enabled:
            fallback_payload = _auto_guarded_sql_fallback(
                question_es=question_es,
                merchant_id=effective_merchant,
            )
            if fallback_payload is not None:
                fallback_payload["planner"] = planner_info
                fallback_payload["failed_query_key"] = resolved_query_key
                return json.dumps(_json_safe(fallback_payload), ensure_ascii=False)

        return json.dumps(
            {
                "ok": False,
                "error": "sql_blocked_or_invalid",
                "intent_id": resolved_intent,
                "query_key": resolved_query_key,
                "resolved_sql": sql,
                "planner": planner_info,
            },
            ensure_ascii=False,
        )

    rows_safe = _json_safe(result.rows)
    facts_payload = _build_facts_payload(resolved_query_key, result.columns, result.rows)

    payload = {
        "ok": True,
        "intent_id": resolved_intent,
        "query_key": resolved_query_key,
        "merchant_id": effective_merchant,
        "resolved_sql": result.sql,
        "params": _json_safe(inferred_slots),
        "summary": result.answer_es,
        "columns": result.columns,
        "rows": rows_safe,
        "row_count": len(result.rows),
        "facts_payload": facts_payload,
        "chart_allowed": bool(entry.get("chart_allowed", False)),
        "planner": planner_info,
    }
    return json.dumps(payload, ensure_ascii=False)


@tool
def run_sql_analytics_tool(
    sql: str,
    merchant_id: str | None = None,
    max_rows: int = 20,
) -> str:
    """Run a guarded read-only SQL query for business analytics.

    Args:
        sql: Candidate SQL statement.
        merchant_id: Merchant override, defaults to configured merchant.
        max_rows: Maximum number of rows to return.

    Returns:
        str: JSON string containing sql, columns, rows, and summary.

    Raises:
        None.
    """
    effective_merchant = (merchant_id or _default_merchant_id()).strip()
    capped_rows = max(1, min(int(max_rows), 50))

    with duckdb.connect(str(_duckdb_path()), read_only=True) as conn:
        result = execute_guarded_sql(
            conn=conn,
            sql=sql,
            merchant_id=effective_merchant,
            max_rows=capped_rows,
        )

    if result is None:
        payload = {
            "ok": False,
            "error": "sql_blocked_or_invalid",
            "hint": ("Usa SELECT/CTE con merchant_id fijo y solo tablas/columnas permitidas."),
        }
        return json.dumps(payload, ensure_ascii=False)

    payload = {
        "ok": True,
        "merchant_id": effective_merchant,
        "sql": result.sql,
        "summary": result.answer_es,
        "columns": result.columns,
        "rows": result.rows,
        "row_count": len(result.rows),
    }
    return json.dumps(payload, ensure_ascii=False)


@tool
def answer_business_question_tool(
    question_es: str,
    merchant_id: str | None = None,
) -> str:
    """Answer common business analytics questions using deterministic handlers.

    Args:
        question_es: User question in Spanish.
        merchant_id: Merchant override, defaults to configured merchant.

    Returns:
        str: JSON string with intent, answer, facts, and fallback flags.

    Raises:
        None.
    """
    effective_merchant = (merchant_id or _default_merchant_id()).strip()
    response = execute_assistant_query(
        AssistantQueryRequest(question_es=question_es, merchant_id=effective_merchant),
        _ASSISTANT_RUNTIME_CONFIG,
    )
    payload = {
        "ok": True,
        "merchant_id": effective_merchant,
        "status": response.status,
        "intent_id": response.intent_id,
        "answer_es": response.answer_es,
        "facts_payload": response.facts_payload,
        "evidence_payload": response.evidence_payload,
    }
    return json.dumps(payload, ensure_ascii=False)


@tool
def assistant_capabilities_tool() -> str:
    """Describe what the assistant can and cannot currently do.

    Args:
        None.

    Returns:
        str: JSON string with supported and unsupported topics.

    Raises:
        None.
    """
    try:
        registry = _load_query_registry()
        queries = registry.get("queries", [])
        if not isinstance(queries, list):
            queries = []

        supported: list[dict[str, Any]] = []
        preview: list[dict[str, Any]] = []
        for entry in queries:
            if not isinstance(entry, dict):
                continue
            item = {
                "intent_id": entry.get("intent_id"),
                "query_key": entry.get("query_key"),
                "description_es": entry.get("description_es"),
                "capability_tags": entry.get("capability_tags", []),
                "chart_allowed": bool(entry.get("chart_allowed", False)),
            }
            status = str(entry.get("status", "")).strip()
            if status == "enabled":
                supported.append(item)
            elif status in {"preview", "disabled"}:
                preview.append(item)

        unsupported = registry.get("unsupported_topics", [])
        if not isinstance(unsupported, list):
            unsupported = []

        payload = {
            "supported": supported,
            "preview": preview,
            "unsupported": unsupported,
            "notes": "Capacidades generadas desde query_registry.yaml.",
        }
    except Exception as exc:
        payload = {
            "supported": [],
            "preview": [],
            "unsupported": [],
            "error": f"capabilities_unavailable: {exc}",
        }
    return json.dumps(payload, ensure_ascii=False)
