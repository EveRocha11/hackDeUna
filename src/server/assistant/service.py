"""Deterministic assistant service backed by DuckDB analytical queries."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb

from server.assistant.answer_generator import generate_answer_from_facts
from server.assistant.intent_classifier import classify_intent_with_llm, load_allowed_intents
from server.assistant.models import AssistantQueryRequest, AssistantQueryResponse
from server.assistant.text2sql import TextToSQLConfig, run_text_to_sql_fallback


@dataclass(frozen=True)
class AssistantRuntimeConfig:
    """Runtime configuration for assistant query execution.

    Args:
        duckdb_path: Path to the analytical DuckDB file.
        end_date: Reference date for period-based calculations.

    Returns:
        AssistantRuntimeConfig: Immutable config object.

    Raises:
        None.
    """

    duckdb_path: Path
    end_date: date
    intent_classifier_enabled: bool
    intent_classifier_model: str
    intent_classifier_temperature: float
    intent_classifier_timeout_seconds: float
    intent_classifier_min_confidence: float
    allowed_intents: tuple[str, ...]
    text2sql: TextToSQLConfig
    answer_generator_enabled: bool
    answer_generator_model: str
    answer_generator_temperature: float
    answer_generator_timeout_seconds: float
    answer_prompt_path: Path


def _env_bool(key: str, default: bool) -> bool:
    """Parse an environment variable as boolean.

    Args:
        key: Environment variable key.
        default: Default value when key is missing.

    Returns:
        bool: Parsed boolean value.

    Raises:
        None.
    """
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_runtime_config() -> AssistantRuntimeConfig:
    """Load assistant runtime config from environment variables.

    Args:
        None.

    Returns:
        AssistantRuntimeConfig: Runtime configuration.

    Raises:
        ValueError: If DATASET_END_DATE has invalid format.
    """
    intents_path = Path(os.getenv("INTENTS_YAML_PATH", "src/agent/semantics/intents.yaml"))
    return AssistantRuntimeConfig(
        duckdb_path=Path(os.getenv("DATA_DUCKDB_PATH", "data/duckdb/analytics.duckdb")),
        end_date=date.fromisoformat(os.getenv("DATASET_END_DATE", "2026-04-18")),
        intent_classifier_enabled=_env_bool("INTENT_CLASSIFIER_ENABLED", True),
        intent_classifier_model=os.getenv("INTENT_CLASSIFIER_MODEL", "gpt-4.1-mini"),
        intent_classifier_temperature=float(os.getenv("INTENT_CLASSIFIER_TEMPERATURE", "0.0")),
        intent_classifier_timeout_seconds=float(
            os.getenv("INTENT_CLASSIFIER_TIMEOUT_SECONDS", "5.0")
        ),
        intent_classifier_min_confidence=float(
            os.getenv("INTENT_CLASSIFIER_MIN_CONFIDENCE", "0.65")
        ),
        allowed_intents=load_allowed_intents(intents_path),
        text2sql=TextToSQLConfig(
            enabled=_env_bool("ENABLE_TEXT2SQL_FALLBACK", False),
            model=os.getenv("TEXT2SQL_MODEL", "gpt-4.1-mini"),
            temperature=float(os.getenv("TEXT2SQL_TEMPERATURE", "0.0")),
            timeout_seconds=float(os.getenv("TEXT2SQL_TIMEOUT_SECONDS", "8.0")),
            max_rows=int(os.getenv("TEXT2SQL_MAX_ROWS", "20")),
        ),
        answer_generator_enabled=_env_bool("ANSWER_GENERATOR_ENABLED", True),
        answer_generator_model=os.getenv("ANSWER_GENERATOR_MODEL", "gpt-4.1-mini"),
        answer_generator_temperature=float(os.getenv("ANSWER_GENERATOR_TEMPERATURE", "0.0")),
        answer_generator_timeout_seconds=float(
            os.getenv("ANSWER_GENERATOR_TIMEOUT_SECONDS", "6.0")
        ),
        answer_prompt_path=Path(
            os.getenv("ANSWER_PROMPT_PATH", "src/agent/prompts/generate_answer.md")
        ),
    )


def _money(value: object) -> float:
    """Round a numeric value to 2 decimals for monetary output.

    Args:
        value: Raw numeric scalar.

    Returns:
        float: Rounded value.

    Raises:
        TypeError: If input value is not numeric.
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"Expected numeric value, got {type(value)!r}")
    return round(float(value), 2)


def _as_float(value: object) -> float:
    """Convert SQL scalar into float.

    Args:
        value: Raw scalar value.

    Returns:
        float: Converted float.

    Raises:
        TypeError: If value is not numeric.
    """
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Expected numeric value, got {type(value)!r}")


def _as_int(value: object) -> int:
    """Convert SQL scalar into int.

    Args:
        value: Raw scalar value.

    Returns:
        int: Converted integer.

    Raises:
        TypeError: If value is not integer-like.
    """
    if isinstance(value, bool):
        raise TypeError("Boolean is not a valid int input")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    raise TypeError(f"Expected integer-like value, got {type(value)!r}")


def _detect_unsupported_topic(question_es: str) -> str | None:
    """Detect deterministic unsupported topics from raw user question.

    Args:
        question_es: User question.

    Returns:
        str | None: Unsupported reason code, or None when supported.

    Raises:
        None.
    """
    question = question_es.strip().lower()
    if any(token in question for token in ["inventario", "stock"]):
        return "inventory_not_in_dataset"
    if any(token in question for token in ["ganancia neta", "utilidad", "margen"]):
        return "profit_not_in_dataset"
    return None


def _question_to_intent_rules(question_es: str) -> str:
    """Map a Spanish question to supported intent id.

    Args:
        question_es: User question.

    Returns:
        str: Detected intent id from rules, or clarification.

    Raises:
        None.
    """
    question = question_es.strip().lower()

    if "mejor" in question and "peor" in question and "día" in question:
        return "best_worst_day"
    if "respecto a la semana pasada" in question or "semana pasada" in question:
        return "income_vs_previous"
    if "mejor o peor" in question and "semana" in question:
        return "income_vs_previous"
    if "nuevos" in question and ("regres" in question or "recurrent" in question):
        return "customer_new_vs_returning"
    if "mejores" in question and "clientes" in question:
        return "top_customers"
    if "dependiendo de pocos clientes" in question:
        return "top_customers"
    if "no volv" in question or "no regres" in question:
        return "inactive_customers"
    if "hora" in question and "vendo" in question:
        return "peak_hours"
    if "vendedores" in question:
        return "seller_performance"
    if "distrib" in question and "ingreso" in question:
        return "income_distribution"
    if "peque" in question and "grande" in question:
        return "income_distribution"
    if "por día" in question and "semana" in question:
        return "income_period"
    if "ayer" in question and "cuánto" in question and "gan" in question:
        return "income_period"
    if "cuánto" in question and "gan" in question:
        return "income_period"

    return "clarification"


def _default_merchant_id(cli_merchant_id: str | None) -> str:
    """Resolve merchant id with environment fallback.

    Args:
        cli_merchant_id: Optional merchant id from request.

    Returns:
        str: Effective merchant id.

    Raises:
        None.
    """
    return (cli_merchant_id or os.getenv("DEUNA_MERCHANT_ID") or "M001").strip()


def _resolve_dates(question_es: str, as_of_date: date) -> tuple[date, date, str]:
    """Resolve date window for the question.

    Args:
        question_es: User question in Spanish.
        as_of_date: Reference date used to compute windows.

    Returns:
        tuple[date, date, str]: Start date, end date, and period label.

    Raises:
        None.
    """
    question = question_es.lower()
    if "ayer" in question:
        yesterday = as_of_date - timedelta(days=1)
        return yesterday, yesterday, "ayer"
    if "mes" in question:
        month_start = as_of_date.replace(day=1)
        return month_start, as_of_date, "mes_actual"

    week_start = as_of_date - timedelta(days=as_of_date.weekday())
    week_end = week_start + timedelta(days=6)
    return week_start, week_end, "semana_actual"


def _build_unsupported_response(
    merchant_id: str,
    reason: str,
) -> AssistantQueryResponse:
    """Build a safe refusal payload for unsupported topics.

    Args:
        merchant_id: Merchant id.
        reason: Unsupported topic reason code.

    Returns:
        AssistantQueryResponse: Unsupported response payload.

    Raises:
        None.
    """
    alternatives = (
        "Puedo ayudarte con ingresos, comparación con semana pasada, clientes, "
        "vendedores u horas pico."
    )
    return AssistantQueryResponse(
        status="unsupported",
        merchant_id=merchant_id,
        intent_id="unsupported",
        answer_es=(f"Ese dato no está en el dataset actual del MVP. {alternatives}"),
        facts_payload={"reason": reason},
        evidence_payload={"policy_key": f"unsupported_topic_{reason}"},
        intent_source="rules_unsupported",
        intent_confidence=1.0,
        chart_allowed=False,
        proactive_flags=[],
        clarification_question_es=None,
    )


def _build_clarification_response(
    merchant_id: str,
    intent_source: str,
    intent_confidence: float | None,
    clarification_question_es: str | None = None,
) -> AssistantQueryResponse:
    """Build a clarification payload for ambiguous queries.

    Args:
        merchant_id: Merchant id.

    Returns:
        AssistantQueryResponse: Clarification response payload.

    Raises:
        None.
    """
    return AssistantQueryResponse(
        status="clarification",
        merchant_id=merchant_id,
        intent_id="clarification",
        answer_es="Necesito un poco más de contexto para ayudarte bien.",
        facts_payload={},
        evidence_payload={},
        intent_source=intent_source,
        intent_confidence=intent_confidence,
        chart_allowed=False,
        proactive_flags=[],
        clarification_question_es=clarification_question_es
        or (
            "¿Quieres ver ingresos de esta semana, comparación con la semana pasada "
            "o clientes nuevos vs recurrentes?"
        ),
    )


def _build_not_implemented_intent_response(
    merchant_id: str,
    intent_id: str,
    intent_source: str,
    intent_confidence: float | None,
) -> AssistantQueryResponse:
    """Build response for known semantic intents not yet implemented in SQL.

    Args:
        merchant_id: Effective merchant id.
        intent_id: Detected supported intent id.
        intent_source: Router source label.
        intent_confidence: Optional confidence score.

    Returns:
        AssistantQueryResponse: Clarification-like response with clear scope.

    Raises:
        None.
    """
    return AssistantQueryResponse(
        status="clarification",
        merchant_id=merchant_id,
        intent_id=intent_id,
        answer_es=(
            "Ese tipo de consulta ya está identificado en la capa semántica, "
            "pero aún no está habilitado en la ejecución SQL de esta fase."
        ),
        facts_payload={},
        evidence_payload={
            "query_key": "intent_not_implemented",
            "params": {"intent_id": intent_id},
        },
        intent_source=intent_source,
        intent_confidence=intent_confidence,
        chart_allowed=False,
        proactive_flags=[],
        clarification_question_es=(
            "Si quieres, te puedo responder ahora ingresos del período, comparación "
            "contra semana pasada o distribución de ingresos por tamaño de venta."
        ),
    )


def _build_text2sql_response(
    *,
    merchant_id: str,
    answer_es: str,
    sql: str,
    columns: list[str],
    rows: list[list[object]],
    intent_source: str,
    intent_confidence: float | None,
) -> AssistantQueryResponse:
    """Build response payload from text-to-SQL fallback execution.

    Args:
        merchant_id: Effective merchant id.
        answer_es: Final short answer in Spanish.
        sql: Executed SQL query.
        columns: Output columns.
        rows: Output rows.
        intent_source: Router source.
        intent_confidence: Optional intent confidence.

    Returns:
        AssistantQueryResponse: Fallback response payload.

    Raises:
        None.
    """
    return AssistantQueryResponse(
        status="ok",
        merchant_id=merchant_id,
        intent_id="text2sql_fallback",
        answer_es=answer_es,
        facts_payload={
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
        },
        evidence_payload={
            "query_key": "text2sql_fallback",
            "sql": sql,
        },
        intent_source=intent_source,
        intent_confidence=intent_confidence,
        chart_allowed=False,
        proactive_flags=[],
        clarification_question_es=None,
    )


def _finalize_answer(
    *,
    request: AssistantQueryRequest,
    config: AssistantRuntimeConfig,
    draft_answer_es: str,
    facts_payload: dict[str, object],
    evidence_payload: dict[str, object],
    proactive_flags: list[str],
) -> str:
    """Compose final user-facing answer from deterministic facts.

    Args:
        request: Original user request.
        config: Runtime configuration.
        draft_answer_es: Deterministic draft answer.
        facts_payload: SQL-grounded facts payload.
        evidence_payload: SQL evidence metadata.
        proactive_flags: Optional proactive flags.

    Returns:
        str: Final Spanish answer.

    Raises:
        None.
    """
    if not config.answer_generator_enabled or not bool(os.getenv("OPENAI_API_KEY")):
        return draft_answer_es

    generated = generate_answer_from_facts(
        question_es=request.question_es,
        draft_answer_es=draft_answer_es,
        facts_payload=facts_payload,
        evidence_payload=evidence_payload,
        proactive_flags=proactive_flags,
        model=config.answer_generator_model,
        temperature=config.answer_generator_temperature,
        timeout_seconds=config.answer_generator_timeout_seconds,
        system_prompt_path=config.answer_prompt_path,
    )
    return generated or draft_answer_es


def _query_one(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    params: list[object],
) -> tuple[object, ...]:
    """Execute SQL expected to return one row.

    Args:
        conn: DuckDB connection.
        sql: SQL query text.
        params: Positional SQL params.

    Returns:
        tuple[object, ...]: Query result row.

    Raises:
        RuntimeError: If query returns no rows.
    """
    row = conn.execute(sql, params).fetchone()
    if row is None:
        raise RuntimeError("Expected one row but got none")
    return row


def execute_assistant_query(
    request: AssistantQueryRequest,
    config: AssistantRuntimeConfig,
) -> AssistantQueryResponse:
    """Resolve one user query into intent, facts, and final answer.

    Args:
        request: Query request payload.
        config: Runtime configuration.

    Returns:
        AssistantQueryResponse: Deterministic query response.

    Raises:
        duckdb.Error: If SQL execution fails.
    """
    merchant_id = _default_merchant_id(request.merchant_id)
    as_of = request.as_of_date or config.end_date
    unsupported_reason = _detect_unsupported_topic(request.question_es)
    intent_source = "rules_fallback"
    intent_confidence: float | None = None
    clarification_hint: str | None = None

    if (
        config.intent_classifier_enabled
        and bool(os.getenv("OPENAI_API_KEY"))
        and config.allowed_intents
    ):
        classification = classify_intent_with_llm(
            question_es=request.question_es,
            allowed_intents=config.allowed_intents,
            model=config.intent_classifier_model,
            temperature=config.intent_classifier_temperature,
            timeout_seconds=config.intent_classifier_timeout_seconds,
        )
        if (
            classification is not None
            and classification.confidence >= config.intent_classifier_min_confidence
        ):
            intent_id = classification.intent_id
            intent_source = "llm"
            intent_confidence = classification.confidence
        else:
            intent_id = _question_to_intent_rules(request.question_es)
            if classification is not None:
                clarification_hint = classification.clarification_question_es
    else:
        intent_id = _question_to_intent_rules(request.question_es)

    if unsupported_reason is not None:
        return _build_unsupported_response(merchant_id, unsupported_reason)
    start_date, end_date, period_key = _resolve_dates(request.question_es, as_of)
    proactive_flags: list[str] = []
    facts_payload: dict[str, object]
    evidence_payload: dict[str, object]
    chart_allowed = False
    answer_es = ""

    with duckdb.connect(str(config.duckdb_path), read_only=True) as conn:
        if intent_id == "clarification":
            text2sql_result = run_text_to_sql_fallback(
                conn=conn,
                question_es=request.question_es,
                merchant_id=merchant_id,
                as_of_date=as_of,
                config=config.text2sql,
            )
            if text2sql_result is not None:
                text2sql_facts: dict[str, object] = {
                    "columns": text2sql_result.columns,
                    "rows": text2sql_result.rows,
                    "row_count": len(text2sql_result.rows),
                }
                text2sql_evidence: dict[str, object] = {
                    "query_key": "text2sql_fallback",
                    "sql": text2sql_result.sql,
                }
                final_answer = _finalize_answer(
                    request=request,
                    config=config,
                    draft_answer_es=text2sql_result.answer_es,
                    facts_payload=text2sql_facts,
                    evidence_payload=text2sql_evidence,
                    proactive_flags=[],
                )
                return _build_text2sql_response(
                    merchant_id=merchant_id,
                    answer_es=final_answer,
                    sql=text2sql_result.sql,
                    columns=text2sql_result.columns,
                    rows=text2sql_result.rows,
                    intent_source="text2sql",
                    intent_confidence=intent_confidence,
                )
            return _build_clarification_response(
                merchant_id=merchant_id,
                intent_source=intent_source,
                intent_confidence=intent_confidence,
                clarification_question_es=clarification_hint,
            )

        if intent_id not in {"income_period", "income_distribution", "income_vs_previous"}:
            text2sql_result = run_text_to_sql_fallback(
                conn=conn,
                question_es=request.question_es,
                merchant_id=merchant_id,
                as_of_date=as_of,
                config=config.text2sql,
            )
            if text2sql_result is not None:
                text2sql_facts: dict[str, object] = {
                    "columns": text2sql_result.columns,
                    "rows": text2sql_result.rows,
                    "row_count": len(text2sql_result.rows),
                }
                text2sql_evidence: dict[str, object] = {
                    "query_key": "text2sql_fallback",
                    "sql": text2sql_result.sql,
                }
                final_answer = _finalize_answer(
                    request=request,
                    config=config,
                    draft_answer_es=text2sql_result.answer_es,
                    facts_payload=text2sql_facts,
                    evidence_payload=text2sql_evidence,
                    proactive_flags=[],
                )
                return _build_text2sql_response(
                    merchant_id=merchant_id,
                    answer_es=final_answer,
                    sql=text2sql_result.sql,
                    columns=text2sql_result.columns,
                    rows=text2sql_result.rows,
                    intent_source="text2sql",
                    intent_confidence=intent_confidence,
                )
            return _build_not_implemented_intent_response(
                merchant_id=merchant_id,
                intent_id=intent_id,
                intent_source=intent_source,
                intent_confidence=intent_confidence,
            )

        if intent_id == "income_period":
            income = _query_one(
                conn,
                """
                SELECT COALESCE(SUM(amount), 0)
                FROM transactions
                WHERE merchant_id = ?
                  AND CAST(occurred_at AS DATE) BETWEEN ? AND ?
                """,
                [merchant_id, start_date.isoformat(), end_date.isoformat()],
            )[0]
            facts_payload = {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "key": period_key,
                },
                "total_income": _money(income),
                "currency": "USD",
            }
            evidence_payload = {
                "query_key": "income_period",
                "params": {
                    "merchant_id": merchant_id,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                },
                "intent_router": {
                    "source": intent_source,
                    "confidence": intent_confidence,
                },
            }
            answer_es = (
                f"Tus ingresos del período {start_date.isoformat()} al "
                f"{end_date.isoformat()} son ${_money(income):.2f}."
            )

        elif intent_id == "income_distribution":
            dist_row = _query_one(
                conn,
                """
                SELECT
                  COUNT(*) AS tx_count,
                  SUM(CASE WHEN amount <= 10 THEN 1 ELSE 0 END) AS small_tx_count,
                  SUM(CASE WHEN amount > 10 AND amount <= 25 THEN 1 ELSE 0 END) AS medium_tx_count,
                  SUM(CASE WHEN amount > 25 THEN 1 ELSE 0 END) AS large_tx_count,
                  COALESCE(
                    SUM(CASE WHEN amount <= 10 THEN amount ELSE 0 END)
                    / NULLIF(SUM(amount), 0),
                    0
                  ) * 100 AS small_income_share_pct,
                  COALESCE(
                    SUM(CASE WHEN amount > 10 AND amount <= 25 THEN amount ELSE 0 END)
                    / NULLIF(SUM(amount), 0),
                    0
                  ) * 100 AS medium_income_share_pct,
                  COALESCE(
                    SUM(CASE WHEN amount > 25 THEN amount ELSE 0 END)
                    / NULLIF(SUM(amount), 0),
                    0
                  ) * 100 AS large_income_share_pct
                FROM transactions
                WHERE merchant_id = ?
                  AND CAST(occurred_at AS DATE) BETWEEN ? AND ?
                """,
                [merchant_id, start_date.isoformat(), end_date.isoformat()],
            )
            tx_count = _as_int(dist_row[0])
            small_tx_count = _as_int(dist_row[1])
            medium_tx_count = _as_int(dist_row[2])
            large_tx_count = _as_int(dist_row[3])
            small_income_share_pct = round(_as_float(dist_row[4]), 2)
            medium_income_share_pct = round(_as_float(dist_row[5]), 2)
            large_income_share_pct = round(_as_float(dist_row[6]), 2)

            if tx_count > 0 and small_tx_count >= max(1, int(0.6 * tx_count)):
                dominant_pattern = "many_small"
            elif large_income_share_pct >= 50:
                dominant_pattern = "few_large"
            else:
                dominant_pattern = "mixed"

            facts_payload = {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "key": period_key,
                },
                "tx_count": tx_count,
                "small_tx_count": small_tx_count,
                "medium_tx_count": medium_tx_count,
                "large_tx_count": large_tx_count,
                "small_income_share_pct": small_income_share_pct,
                "medium_income_share_pct": medium_income_share_pct,
                "large_income_share_pct": large_income_share_pct,
                "dominant_pattern": dominant_pattern,
                "currency": "USD",
            }
            evidence_payload = {
                "query_key": "income_distribution",
                "params": {
                    "merchant_id": merchant_id,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                },
                "intent_router": {
                    "source": intent_source,
                    "confidence": intent_confidence,
                },
            }
            chart_allowed = True
            answer_es = (
                "En este período, tus ingresos vienen "
                f"{small_income_share_pct:.2f}% de ventas pequeñas, "
                f"{medium_income_share_pct:.2f}% de medianas y "
                f"{large_income_share_pct:.2f}% de grandes."
            )

        elif intent_id == "income_vs_previous":
            current_start = start_date
            current_end = end_date
            prev_start = current_start - timedelta(days=7)
            prev_end = current_end - timedelta(days=7)
            compare_row = _query_one(
                conn,
                """
                WITH current_p AS (
                  SELECT COALESCE(SUM(amount), 0) AS income
                  FROM transactions
                  WHERE merchant_id = ?
                    AND CAST(occurred_at AS DATE) BETWEEN ? AND ?
                ), previous_p AS (
                  SELECT COALESCE(SUM(amount), 0) AS income
                  FROM transactions
                  WHERE merchant_id = ?
                    AND CAST(occurred_at AS DATE) BETWEEN ? AND ?
                )
                SELECT
                  current_p.income,
                  previous_p.income,
                  CASE
                    WHEN previous_p.income = 0 THEN NULL
                    ELSE ((current_p.income - previous_p.income) / previous_p.income) * 100
                  END AS pct_change
                FROM current_p, previous_p
                """,
                [
                    merchant_id,
                    current_start.isoformat(),
                    current_end.isoformat(),
                    merchant_id,
                    prev_start.isoformat(),
                    prev_end.isoformat(),
                ],
            )
            pct_change = None if compare_row[2] is None else round(_as_float(compare_row[2]), 2)
            if pct_change is not None and pct_change <= -10:
                proactive_flags.append("sales_drop_alert")

            facts_payload = {
                "period": {
                    "current_start": current_start.isoformat(),
                    "current_end": current_end.isoformat(),
                    "previous_start": prev_start.isoformat(),
                    "previous_end": prev_end.isoformat(),
                },
                "current_period_income": _money(compare_row[0]),
                "previous_period_income": _money(compare_row[1]),
                "pct_change": pct_change,
                "currency": "USD",
            }
            evidence_payload = {
                "query_key": "income_vs_previous",
                "params": {
                    "merchant_id": merchant_id,
                    "current_start": current_start.isoformat(),
                    "current_end": current_end.isoformat(),
                    "previous_start": prev_start.isoformat(),
                    "previous_end": prev_end.isoformat(),
                },
                "intent_router": {
                    "source": intent_source,
                    "confidence": intent_confidence,
                },
            }

            if pct_change is None:
                answer_es = (
                    "No puedo calcular variación porcentual porque el período anterior "
                    "no tiene ingresos."
                )
            else:
                direction = "subieron" if pct_change >= 0 else "bajaron"
                answer_es = (
                    "Comparado con la semana anterior, tus ingresos "
                    f"{direction} {abs(pct_change):.2f}% "
                    f"(${_money(compare_row[1]):.2f} -> ${_money(compare_row[0]):.2f})."
                )
                if "sales_drop_alert" in proactive_flags:
                    answer_es += " Te sugiero revisar horas pico y vendedores primero."

        else:
            return _build_clarification_response(
                merchant_id=merchant_id,
                intent_source=intent_source,
                intent_confidence=intent_confidence,
            )

    return AssistantQueryResponse(
        status="ok",
        merchant_id=merchant_id,
        intent_id=intent_id,
        answer_es=_finalize_answer(
            request=request,
            config=config,
            draft_answer_es=answer_es,
            facts_payload=facts_payload,
            evidence_payload=evidence_payload,
            proactive_flags=proactive_flags,
        ),
        facts_payload=facts_payload,
        evidence_payload=evidence_payload,
        intent_source=intent_source,
        intent_confidence=intent_confidence,
        chart_allowed=chart_allowed,
        proactive_flags=proactive_flags,
        clarification_question_es=None,
    )
