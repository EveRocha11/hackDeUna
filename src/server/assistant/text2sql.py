"""Guarded text-to-SQL fallback utilities for long-tail analytical questions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

import duckdb
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class TextToSQLConfig:
    """Configuration for guarded text-to-SQL fallback.

    Args:
        enabled: Enable text-to-SQL fallback.
        model: LLM model name used for SQL synthesis.
        temperature: Sampling temperature.
        timeout_seconds: Timeout for model calls.
        max_rows: Max rows returned from query execution.

    Returns:
        TextToSQLConfig: Immutable config object.

    Raises:
        None.
    """

    enabled: bool
    model: str
    temperature: float
    timeout_seconds: float
    max_rows: int


@dataclass(frozen=True)
class TextToSQLResult:
    """Execution result of text-to-SQL fallback.

    Args:
        sql: Final validated SQL statement.
        answer_es: Natural language answer.
        rows: Tabular rows returned by execution.
        columns: Column names returned by execution.

    Returns:
        TextToSQLResult: Structured result payload.

    Raises:
        None.
    """

    sql: str
    answer_es: str
    rows: list[list[object]]
    columns: list[str]


class SQLDraft(BaseModel):
    """Structured output schema for SQL synthesis.

    Args:
        sql: Candidate SQL query.
        rationale_es: Short rationale in Spanish.

    Returns:
        SQLDraft: Validated draft object.

    Raises:
        ValueError: If required fields are invalid.
    """

    sql: str = Field(description="Single SELECT statement compatible with DuckDB")
    rationale_es: str = Field(description="Short rationale in Spanish")


ALLOWED_TABLES: tuple[str, ...] = (
    "transactions",
    "customers",
    "sellers",
    "merchants",
)

ALLOWED_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "merchant_id",
    "seller_id",
    "customer_id",
    "occurred_at",
    "amount",
    "currency",
    "payer_doc_masked",
    "payer_account_masked",
    "payment_channel",
    "week_start_day",
    "tx_sequence",
    "customer_display_name",
    "frequency_profile",
    "first_seen_at",
    "id_doc_masked",
    "account_masked",
    "role",
    "seller_display_name",
    "merchant_name",
    "city",
    "category",
    "created_at",
)

BLOCKED_SQL_KEYWORDS: tuple[str, ...] = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "truncate",
    "grant",
    "revoke",
    "attach",
    "detach",
    "copy",
    "call",
    "pragma",
)

ALLOWED_FUNCTIONS: tuple[str, ...] = (
    "sum",
    "count",
    "avg",
    "min",
    "max",
    "coalesce",
    "nullif",
    "cast",
    "extract",
    "round",
    "date_trunc",
    "lower",
    "upper",
    "abs",
)


def _normalize_sql(sql: str) -> str:
    """Normalize SQL spacing for deterministic validation and execution.

    Args:
        sql: Raw SQL string.

    Returns:
        str: Normalized SQL string.

    Raises:
        None.
    """
    compact = " ".join(sql.strip().split())
    return compact.rstrip(";")


def _repair_occurred_at_date_comparisons(sql: str) -> str:
    """Repair common occurred_at vs DATE comparison issue for DuckDB schema.

    Args:
        sql: Candidate SQL string.

    Returns:
        str: SQL string with safe occurred_at date casts when needed.

    Raises:
        None.
    """
    lowered = sql.lower()
    if "occurred_at" not in lowered:
        return sql
    if "cast(occurred_at as date)" in lowered:
        return sql
    if "date '" not in lowered:
        return sql
    return re.sub(r"\boccurred_at\b", "CAST(occurred_at AS DATE)", sql, flags=re.IGNORECASE)


def _validate_sql(sql: str, merchant_id: str) -> bool:
    """Validate SQL with hard guardrails.

    Args:
        sql: Candidate SQL.
        merchant_id: Active merchant id that must be constrained.

    Returns:
        bool: True when SQL passes validation.

    Raises:
        None.
    """
    lowered = sql.lower()

    if ";" in sql:
        return False
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return False
    if any(re.search(rf"\b{kw}\b", lowered) for kw in BLOCKED_SQL_KEYWORDS):
        return False

    if "merchant_id" not in lowered:
        return False
    if merchant_id.lower() not in lowered:
        return False

    cte_names = set(re.findall(r"\bwith\s+([a-z_][a-z0-9_]*)\s+as\b", lowered))
    cte_names |= set(re.findall(r",\s*([a-z_][a-z0-9_]*)\s+as\b", lowered))

    used_tables = set(re.findall(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)\b", lowered))
    if not used_tables:
        return False
    if any(table not in ALLOWED_TABLES and table not in cte_names for table in used_tables):
        return False

    if re.search(r"--|/\*|\*/", lowered):
        return False

    # Basic safe character filter to reduce malformed or injected SQL shapes.
    if re.search(r"[^a-z0-9_\s\.,\(\)\*=<>!\'\-\+/%]", lowered):
        return False

    # Validate function calls belong to a known analytical set.
    called_functions = re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", lowered)
    for func in called_functions:
        if func not in ALLOWED_FUNCTIONS and func not in {"select", "with", "as"}:
            return False

    return True


def _build_sql_prompt(question_es: str, merchant_id: str, as_of_date: date, max_rows: int) -> str:
    """Build prompt for SQL synthesis constrained to known schema.

    Args:
        question_es: User question in Spanish.
        merchant_id: Active merchant id.
        as_of_date: Reference date.
        max_rows: Max rows allowed in output.

    Returns:
        str: Prompt text.

    Raises:
        None.
    """
    allowed_tables = ", ".join(ALLOWED_TABLES)
    allowed_columns = ", ".join(ALLOWED_COLUMNS)
    return (
        "Genera SQL de DuckDB para responder la pregunta de un microcomercio.\n"
        "Reglas estrictas:\n"
        "- Un solo statement.\n"
        "- Solo SELECT o WITH ... SELECT.\n"
        "- Solo tablas permitidas.\n"
        "- Siempre filtra por merchant_id exacto.\n"
        f"- merchant_id obligatorio: '{merchant_id}'.\n"
        f"- Fecha de referencia para expresiones relativas: {as_of_date.isoformat()}.\n"
        f"- Incluye LIMIT <= {max_rows} si hay salida tabular.\n"
        "- occurred_at es VARCHAR: si comparas con fechas usa CAST(occurred_at AS DATE).\n"
        "- No uses tablas ni columnas fuera del catálogo.\n"
        f"Tablas permitidas: {allowed_tables}.\n"
        f"Columnas permitidas: {allowed_columns}.\n"
        f"Pregunta: {question_es}"
    )


def run_text_to_sql_fallback(
    *,
    conn: duckdb.DuckDBPyConnection,
    question_es: str,
    merchant_id: str,
    as_of_date: date,
    config: TextToSQLConfig,
) -> TextToSQLResult | None:
    """Try to answer a question via guarded text-to-SQL fallback.

    Args:
        conn: Active DuckDB read-only connection.
        question_es: User question in Spanish.
        merchant_id: Active merchant id.
        as_of_date: Reference date.
        config: Text-to-SQL runtime configuration.

    Returns:
        TextToSQLResult | None: Result when successful, else None.

    Raises:
        None.
    """
    if not config.enabled:
        return None

    client = ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
        timeout=config.timeout_seconds,
    )

    prompt = _build_sql_prompt(
        question_es=question_es,
        merchant_id=merchant_id,
        as_of_date=as_of_date,
        max_rows=config.max_rows,
    )

    try:
        structured = client.with_structured_output(SQLDraft)
        raw = structured.invoke(prompt)
        draft = SQLDraft.model_validate(raw) if isinstance(raw, dict) else raw
    except Exception:
        return None

    sql = _normalize_sql(draft.sql)
    if not _validate_sql(sql, merchant_id):
        return None

    try:
        rows = conn.execute(sql).fetchmany(config.max_rows)
        columns = [desc[0] for desc in (conn.description or [])]
    except Exception:
        repaired_sql = _normalize_sql(_repair_occurred_at_date_comparisons(sql))
        if repaired_sql == sql or not _validate_sql(repaired_sql, merchant_id):
            return None
        try:
            rows = conn.execute(repaired_sql).fetchmany(config.max_rows)
            columns = [desc[0] for desc in (conn.description or [])]
            sql = repaired_sql
        except Exception:
            return None

    if not rows:
        answer = "No encontré datos para esa consulta en el período o filtros solicitados."
    elif len(rows) == 1 and len(rows[0]) == 1:
        value = rows[0][0]
        answer = f"El resultado es: {value}."
    else:
        answer = (
            f"Encontré {len(rows)} filas relevantes. Te comparto una vista resumida "
            "en el payload estructurado."
        )

    return TextToSQLResult(
        sql=sql,
        answer_es=answer,
        rows=[list(row) for row in rows],
        columns=columns,
    )
