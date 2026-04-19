"""Build a deterministic MVP evaluation set grounded on the generated dataset."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import yaml


@dataclass(frozen=True)
class EvalBuildConfig:
    """Runtime configuration for evaluation set construction.

    Args:
        duckdb_path: Path to analytical DuckDB database.
        output_path: JSON output file path for evaluation items.
        merchant_id: Active merchant for single-merchant evaluation mode.
        end_date: Reference end date used by period-based questions.

    Returns:
        EvalBuildConfig: Immutable config object.

    Raises:
        ValueError: If merchant_id is empty.
    """

    duckdb_path: Path
    output_path: Path
    merchant_id: str
    end_date: date
    questions_bank_path: Path
    profiles_path: Path
    question_refine_model: str
    question_refine_temperature: float
    refine_questions_with_llm: bool

    def __post_init__(self) -> None:
        """Validate config values.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If merchant_id is invalid.
        """
        if not self.merchant_id:
            raise ValueError("merchant_id cannot be empty")


def _load_config(cli_merchant_id: str | None) -> EvalBuildConfig:
    """Load evaluation build config from env with optional CLI override.

    Args:
        cli_merchant_id: Optional merchant override from CLI.

    Returns:
        EvalBuildConfig: Parsed configuration.

    Raises:
        ValueError: If date parsing fails.
    """
    merchant_id = (
        cli_merchant_id or os.getenv("EVAL_MERCHANT_ID") or os.getenv("DEUNA_MERCHANT_ID", "M001")
    )
    end_date = date.fromisoformat(os.getenv("DATASET_END_DATE", "2026-04-18"))

    return EvalBuildConfig(
        duckdb_path=Path(os.getenv("DATA_DUCKDB_PATH", "data/duckdb/analytics.duckdb")),
        output_path=Path("data/evals/eval_set_v1.json"),
        questions_bank_path=Path("src/agent/semantics/eval_question_bank.yaml"),
        profiles_path=Path(os.getenv("PROFILES_PATH", "src/agent/semantics/profiles.yaml")),
        question_refine_model=os.getenv("QUESTION_REFINE_MODEL", "gpt-4o-mini"),
        question_refine_temperature=float(os.getenv("QUESTION_REFINE_TEMPERATURE", "0.0")),
        refine_questions_with_llm=_env_bool("REFINE_QUESTIONS_WITH_LLM", True),
        merchant_id=merchant_id,
        end_date=end_date,
    )


def _env_bool(key: str, default: bool) -> bool:
    """Parse an environment variable as boolean.

    Args:
        key: Environment variable name.
        default: Value used when variable is missing.

    Returns:
        bool: Parsed value.

    Raises:
        None.
    """
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_questions_bank(path: Path) -> dict[str, dict[str, str]]:
    """Load eval question/profile mapping from YAML.

    Args:
        path: YAML file path.

    Returns:
        dict[str, dict[str, str]]: Mapping by eval_id with profile_id and question_es.

    Raises:
        FileNotFoundError: If YAML file does not exist.
        ValueError: If required keys are missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Questions bank file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Questions bank must be a mapping")

    items = raw.get("eval_items")
    if not isinstance(items, list):
        raise ValueError("Questions bank must include an 'eval_items' list")

    catalog: dict[str, dict[str, str]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each eval item must be a mapping")
        eval_id = str(item.get("eval_id", "")).strip()
        profile_id = str(item.get("profile_id", "")).strip()
        question_es = str(item.get("question_es", "")).strip()
        if not eval_id or not profile_id or not question_es:
            raise ValueError("Each eval item must include eval_id, profile_id, question_es")
        catalog[eval_id] = {"profile_id": profile_id, "question_es": question_es}

    return catalog


def _meta_for(catalog: dict[str, dict[str, str]], eval_id: str) -> tuple[str, str]:
    """Get profile and question text for an eval item.

    Args:
        catalog: Eval metadata mapping.
        eval_id: Evaluation ID.

    Returns:
        tuple[str, str]: (profile_id, question_es)

    Raises:
        KeyError: If eval_id is not present in catalog.
    """
    if eval_id not in catalog:
        raise KeyError(f"Missing eval metadata for {eval_id}")
    meta = catalog[eval_id]
    return meta["profile_id"], meta["question_es"]


def _load_profiles_payload(path: Path) -> dict[str, Any]:
    """Load profiles and global question requirements.

    Args:
        path: YAML profiles file path.

    Returns:
        dict[str, Any]: Parsed payload with profile and rule context.

    Raises:
        FileNotFoundError: If profiles file is missing.
        ValueError: If YAML structure is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Profiles file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Profiles payload must be a mapping")

    return {
        "profiles": raw.get("profiles", {}),
        "global_question_requirements": raw.get("global_question_requirements", {}),
    }


class QuestionRewriteResult(BaseModel):
    """Structured output schema for refined question text.

    Args:
        question_es: Refined Spanish question.

    Returns:
        QuestionRewriteResult: Parsed structured response.

    Raises:
        None.
    """

    question_es: str = Field(description="Refined question in Spanish")


def _refine_question_with_profile_context(
    question_es: str,
    profile_id: str,
    profiles_payload: dict[str, Any],
    config: EvalBuildConfig,
    client: ChatOpenAI,
) -> str:
    """Refine one question with profile-aware style constraints.

    Args:
        question_es: Base question text.
        profile_id: Profile key for style context.
        profiles_payload: Profiles and global style rules.
        config: Eval build configuration.
        client: ChatOpenAI client.

    Returns:
        str: Refined question, or original question if LLM fails.

    Raises:
        None.
    """
    profiles = profiles_payload.get("profiles", {})
    requirements = profiles_payload.get("global_question_requirements", {})
    if not isinstance(profiles, dict) or not isinstance(requirements, dict):
        return question_es

    profile_meta = profiles.get(profile_id, {})
    if not isinstance(profile_meta, dict):
        profile_meta = {}

    style_notes = profile_meta.get("style_notes", [])
    style_text = (
        "\n".join(f"- {str(note)}" for note in style_notes) if isinstance(style_notes, list) else ""
    )

    avoid_terms = requirements.get("avoid_terms", [])
    avoid_text = (
        ", ".join(str(term) for term in avoid_terms) if isinstance(avoid_terms, list) else ""
    )

    rules = requirements.get("rules", [])
    rules_text = "\n".join(f"- {str(rule)}" for rule in rules) if isinstance(rules, list) else ""

    prompt = (
        "Refina una pregunta para un microcomercio en Ecuador.\n"
        "Mantén exactamente la misma intención analítica y período temporal.\n"
        "Solo mejora lenguaje para que suene más natural para el perfil.\n"
        "No cambies la semántica de fondo.\n"
        "No agregues temas fuera de alcance.\n\n"
        f"Pregunta base: {question_es}\n"
        f"Perfil: {profile_meta.get('label', profile_id)}\n"
        f"Contexto: {profile_meta.get('context', '')}\n"
        f"Objetivo del perfil: {profile_meta.get('objective', '')}\n"
        f"Notas de estilo:\n{style_text}\n"
        f"Tono global: {requirements.get('tone', 'informal, simple, cercano')}\n"
        f"Evitar términos: {avoid_text}\n"
        f"Reglas:\n{rules_text}\n"
    )

    try:
        structured = client.with_structured_output(QuestionRewriteResult)
        result = structured.invoke(prompt)
        if isinstance(result, dict):
            refined_raw = str(result.get("question_es", ""))
        else:
            refined_raw = result.question_es
        refined = refined_raw.strip()
        return refined if refined else question_es
    except Exception:
        return question_es


def _refine_question_catalog_with_llm(
    catalog: dict[str, dict[str, str]],
    profiles_payload: dict[str, Any],
    config: EvalBuildConfig,
) -> dict[str, dict[str, str]]:
    """Refine all catalog questions with profile context using LLM.

    Args:
        catalog: Base eval question catalog.
        profiles_payload: Profiles and global style rules.
        config: Eval build configuration.

    Returns:
        dict[str, dict[str, str]]: Catalog with refined question text.

    Raises:
        None.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return catalog

    client = ChatOpenAI(
        model=config.question_refine_model,
        temperature=config.question_refine_temperature,
    )

    refined_catalog: dict[str, dict[str, str]] = {}
    for eval_id, meta in catalog.items():
        profile_id = meta["profile_id"]
        if eval_id in {"E16", "E17"}:
            refined_catalog[eval_id] = {
                "profile_id": profile_id,
                "question_es": meta["question_es"],
            }
            continue
        refined_question = _refine_question_with_profile_context(
            question_es=meta["question_es"],
            profile_id=profile_id,
            profiles_payload=profiles_payload,
            config=config,
            client=client,
        )
        refined_catalog[eval_id] = {"profile_id": profile_id, "question_es": refined_question}

    return refined_catalog


def _money(value: float) -> float:
    """Round currency-like values for stable expected facts.

    Args:
        value: Numeric value.

    Returns:
        float: Value rounded to 2 decimals.

    Raises:
        None.
    """
    return round(float(value), 2)


def _as_float(value: object) -> float:
    """Convert query output scalar to float.

    Args:
        value: Raw scalar value.

    Returns:
        float: Converted numeric value.

    Raises:
        TypeError: If value is not numeric.
    """
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Expected numeric value, got: {type(value)!r}")


def _as_int(value: object) -> int:
    """Convert query output scalar to int.

    Args:
        value: Raw scalar value.

    Returns:
        int: Converted integer value.

    Raises:
        TypeError: If value is not integer-compatible.
    """
    if isinstance(value, bool):
        raise TypeError("Boolean is not a valid int input")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    raise TypeError(f"Expected integer-like value, got: {type(value)!r}")


def _query_one(
    conn: duckdb.DuckDBPyConnection, sql: str, params: list[object]
) -> tuple[object, ...]:
    """Execute a query expected to return one row.

    Args:
        conn: DuckDB connection.
        sql: SQL statement.
        params: SQL parameters.

    Returns:
        tuple[object, ...]: First row tuple.

    Raises:
        RuntimeError: If query returns no rows.
    """
    row = conn.execute(sql, params).fetchone()
    if row is None:
        raise RuntimeError("Expected one row but got none")
    return row


def _build_items(config: EvalBuildConfig) -> list[dict[str, object]]:
    """Construct grounded evaluation items from deterministic SQL facts.

    Args:
        config: Evaluation build configuration.

    Returns:
        list[dict[str, object]]: Evaluation item payload list.

    Raises:
        duckdb.Error: If SQL execution fails.
    """
    merchant_id = config.merchant_id
    week_start = config.end_date - timedelta(days=config.end_date.weekday())
    week_end = week_start + timedelta(days=6)
    prev_week_start = week_start - timedelta(days=7)
    prev_week_end = week_end - timedelta(days=7)
    month_start = config.end_date.replace(day=1)
    question_catalog = _load_questions_bank(config.questions_bank_path)
    if config.refine_questions_with_llm:
        profiles_payload = _load_profiles_payload(config.profiles_path)
        question_catalog = _refine_question_catalog_with_llm(
            question_catalog, profiles_payload, config
        )

    items: list[dict[str, object]] = []

    with duckdb.connect(str(config.duckdb_path), read_only=True) as conn:
        # 1) Income this week
        income_week = _query_one(
            conn,
            """
            SELECT COALESCE(SUM(amount), 0)
            FROM transactions
            WHERE merchant_id = ?
              AND CAST(occurred_at AS DATE) BETWEEN ? AND ?
            """,
            [merchant_id, week_start.isoformat(), week_end.isoformat()],
        )[0]
        items.append(
            {
                "eval_id": "E01",
                "profile_id": _meta_for(question_catalog, "E01")[0],
                "persona_id": _meta_for(question_catalog, "E01")[0],
                "merchant_id": merchant_id,
                "question_es": _meta_for(question_catalog, "E01")[1],
                "expected_intent_id": "income_period",
                "chart_allowed": False,
                "proactive_expected": False,
                "expected_facts": {
                    "period": {"start": week_start.isoformat(), "end": week_end.isoformat()},
                    "total_income": _money(_as_float(income_week)),
                    "currency": "USD",
                },
                "expected_evidence": {
                    "query_key": "income_period_week",
                    "params": {
                        "merchant_id": merchant_id,
                        "start_date": week_start.isoformat(),
                        "end_date": week_end.isoformat(),
                    },
                },
                "pass_fail_criteria": {
                    "numeric_tolerance": 0.01,
                    "must_include_period": True,
                    "must_not_hallucinate": True,
                },
            }
        )

        # 2) Income distribution by ticket size this week
        dist_row = _query_one(
            conn,
            """
            SELECT
              COUNT(*) AS tx_count,
              SUM(CASE WHEN amount <= 10 THEN 1 ELSE 0 END) AS small_tx_count,
              SUM(CASE WHEN amount > 10 AND amount <= 25 THEN 1 ELSE 0 END) AS medium_tx_count,
              SUM(CASE WHEN amount > 25 THEN 1 ELSE 0 END) AS large_tx_count,
              COALESCE(SUM(CASE WHEN amount <= 10 THEN amount ELSE 0 END) / NULLIF(SUM(amount), 0), 0) * 100 AS small_income_share_pct,
              COALESCE(SUM(CASE WHEN amount > 10 AND amount <= 25 THEN amount ELSE 0 END) / NULLIF(SUM(amount), 0), 0) * 100 AS medium_income_share_pct,
              COALESCE(SUM(CASE WHEN amount > 25 THEN amount ELSE 0 END) / NULLIF(SUM(amount), 0), 0) * 100 AS large_income_share_pct
            FROM transactions
            WHERE merchant_id = ?
              AND CAST(occurred_at AS DATE) BETWEEN ? AND ?
            """,
            [merchant_id, week_start.isoformat(), week_end.isoformat()],
        )
        tx_count_week = _as_int(dist_row[0])
        small_tx_count = _as_int(dist_row[1])
        medium_tx_count = _as_int(dist_row[2])
        large_tx_count = _as_int(dist_row[3])
        small_income_share_pct = round(_as_float(dist_row[4]), 2)
        medium_income_share_pct = round(_as_float(dist_row[5]), 2)
        large_income_share_pct = round(_as_float(dist_row[6]), 2)

        if small_tx_count >= max(1, int(0.6 * tx_count_week)):
            dominant_pattern = "many_small"
        elif large_income_share_pct >= 50.0:
            dominant_pattern = "few_large"
        else:
            dominant_pattern = "mixed"

        items.append(
            {
                "eval_id": "E02",
                "profile_id": _meta_for(question_catalog, "E02")[0],
                "persona_id": _meta_for(question_catalog, "E02")[0],
                "merchant_id": merchant_id,
                "question_es": _meta_for(question_catalog, "E02")[1],
                "expected_intent_id": "income_distribution",
                "chart_allowed": False,
                "proactive_expected": False,
                "expected_facts": {
                    "period": {"start": week_start.isoformat(), "end": week_end.isoformat()},
                    "tx_count": tx_count_week,
                    "small_tx_count": small_tx_count,
                    "medium_tx_count": medium_tx_count,
                    "large_tx_count": large_tx_count,
                    "small_income_share_pct": small_income_share_pct,
                    "medium_income_share_pct": medium_income_share_pct,
                    "large_income_share_pct": large_income_share_pct,
                    "dominant_pattern": dominant_pattern,
                    "currency": "USD",
                },
                "expected_evidence": {
                    "query_key": "income_distribution_week",
                    "params": {
                        "merchant_id": merchant_id,
                        "start_date": week_start.isoformat(),
                        "end_date": week_end.isoformat(),
                    },
                },
                "pass_fail_criteria": {
                    "numeric_tolerance": 0.02,
                    "must_not_hallucinate": True,
                },
            }
        )

        # 3) Income yesterday
        yesterday = config.end_date - timedelta(days=1)
        income_yesterday = _query_one(
            conn,
            """
            SELECT COALESCE(SUM(amount), 0)
            FROM transactions
            WHERE merchant_id = ?
              AND CAST(occurred_at AS DATE) = ?
            """,
            [merchant_id, yesterday.isoformat()],
        )[0]
        items.append(
            {
                "eval_id": "E03",
                "profile_id": _meta_for(question_catalog, "E03")[0],
                "persona_id": _meta_for(question_catalog, "E03")[0],
                "merchant_id": merchant_id,
                "question_es": _meta_for(question_catalog, "E03")[1],
                "expected_intent_id": "income_period",
                "chart_allowed": False,
                "proactive_expected": False,
                "expected_facts": {
                    "date": yesterday.isoformat(),
                    "total_income": _money(_as_float(income_yesterday)),
                    "currency": "USD",
                },
                "expected_evidence": {
                    "query_key": "income_period_yesterday",
                    "params": {"merchant_id": merchant_id, "date": yesterday.isoformat()},
                },
                "pass_fail_criteria": {
                    "numeric_tolerance": 0.01,
                    "must_include_date": True,
                    "must_not_hallucinate": True,
                },
            }
        )

        # 4) Best and worst day this week
        best_worst_row = _query_one(
            conn,
            """
            WITH daily AS (
              SELECT CAST(occurred_at AS DATE) AS d, SUM(amount) AS income
              FROM transactions
              WHERE merchant_id = ?
                AND CAST(occurred_at AS DATE) BETWEEN ? AND ?
              GROUP BY 1
            )
            SELECT
              (SELECT d FROM daily ORDER BY income DESC, d ASC LIMIT 1) AS best_day,
              (SELECT income FROM daily ORDER BY income DESC, d ASC LIMIT 1) AS best_income,
              (SELECT d FROM daily ORDER BY income ASC, d ASC LIMIT 1) AS worst_day,
              (SELECT income FROM daily ORDER BY income ASC, d ASC LIMIT 1) AS worst_income
            """,
            [merchant_id, week_start.isoformat(), week_end.isoformat()],
        )
        items.append(
            {
                "eval_id": "E04",
                "profile_id": _meta_for(question_catalog, "E04")[0],
                "persona_id": _meta_for(question_catalog, "E04")[0],
                "merchant_id": merchant_id,
                "question_es": _meta_for(question_catalog, "E04")[1],
                "expected_intent_id": "best_worst_day",
                "chart_allowed": False,
                "proactive_expected": False,
                "expected_facts": {
                    "best_day": str(best_worst_row[0]),
                    "best_day_income": _money(_as_float(best_worst_row[1])),
                    "worst_day": str(best_worst_row[2]),
                    "worst_day_income": _money(_as_float(best_worst_row[3])),
                    "currency": "USD",
                },
                "expected_evidence": {
                    "query_key": "best_worst_day_week",
                    "params": {
                        "merchant_id": merchant_id,
                        "start_date": week_start.isoformat(),
                        "end_date": week_end.isoformat(),
                    },
                },
                "pass_fail_criteria": {
                    "exact_date_match": True,
                    "numeric_tolerance": 0.01,
                },
            }
        )

        # 5) Income this week vs previous week
        compare_row = _query_one(
            conn,
            """
            WITH current_p AS (
              SELECT COALESCE(SUM(amount), 0) AS income
              FROM transactions
              WHERE merchant_id = ? AND CAST(occurred_at AS DATE) BETWEEN ? AND ?
            ), previous_p AS (
              SELECT COALESCE(SUM(amount), 0) AS income
              FROM transactions
              WHERE merchant_id = ? AND CAST(occurred_at AS DATE) BETWEEN ? AND ?
            )
            SELECT
              current_p.income,
              previous_p.income,
              CASE WHEN previous_p.income = 0 THEN NULL ELSE ((current_p.income - previous_p.income) / previous_p.income) * 100 END AS pct_change
            FROM current_p, previous_p
            """,
            [
                merchant_id,
                week_start.isoformat(),
                week_end.isoformat(),
                merchant_id,
                prev_week_start.isoformat(),
                prev_week_end.isoformat(),
            ],
        )
        pct_change = None if compare_row[2] is None else round(_as_float(compare_row[2]), 2)
        items.append(
            {
                "eval_id": "E05",
                "profile_id": _meta_for(question_catalog, "E05")[0],
                "persona_id": _meta_for(question_catalog, "E05")[0],
                "merchant_id": merchant_id,
                "question_es": _meta_for(question_catalog, "E05")[1],
                "expected_intent_id": "income_vs_previous",
                "chart_allowed": False,
                "proactive_expected": bool(pct_change is not None and pct_change <= -10.0),
                "expected_facts": {
                    "current_period_income": _money(_as_float(compare_row[0])),
                    "previous_period_income": _money(_as_float(compare_row[1])),
                    "pct_change": pct_change,
                    "currency": "USD",
                },
                "expected_evidence": {
                    "query_key": "income_vs_previous_week",
                    "params": {
                        "merchant_id": merchant_id,
                        "current_start": week_start.isoformat(),
                        "current_end": week_end.isoformat(),
                        "previous_start": prev_week_start.isoformat(),
                        "previous_end": prev_week_end.isoformat(),
                    },
                },
                "pass_fail_criteria": {
                    "numeric_tolerance": 0.02,
                    "must_include_direction": True,
                },
            }
        )

        # 6) New vs returning customers this month
        customer_row = _query_one(
            conn,
            """
            WITH month_tx AS (
              SELECT customer_id
              FROM transactions
              WHERE merchant_id = ?
                AND CAST(occurred_at AS DATE) BETWEEN ? AND ?
            ), first_seen AS (
              SELECT customer_id, MIN(CAST(occurred_at AS DATE)) AS first_purchase
              FROM transactions
              WHERE merchant_id = ?
              GROUP BY customer_id
            )
            SELECT
              COUNT(DISTINCT CASE WHEN first_purchase BETWEEN ? AND ? THEN month_tx.customer_id END) AS new_customers,
              COUNT(DISTINCT CASE WHEN first_purchase < ? THEN month_tx.customer_id END) AS returning_customers,
              COUNT(DISTINCT month_tx.customer_id) AS unique_customers
            FROM month_tx
            JOIN first_seen USING (customer_id)
            """,
            [
                merchant_id,
                month_start.isoformat(),
                config.end_date.isoformat(),
                merchant_id,
                month_start.isoformat(),
                config.end_date.isoformat(),
                month_start.isoformat(),
            ],
        )
        unique_customers = _as_int(customer_row[2])
        returning_share = (
            0.0
            if unique_customers == 0
            else round((_as_int(customer_row[1]) / unique_customers) * 100, 2)
        )
        items.append(
            {
                "eval_id": "E06",
                "profile_id": _meta_for(question_catalog, "E06")[0],
                "persona_id": _meta_for(question_catalog, "E06")[0],
                "merchant_id": merchant_id,
                "question_es": _meta_for(question_catalog, "E06")[1],
                "expected_intent_id": "customer_new_vs_returning",
                "chart_allowed": False,
                "proactive_expected": False,
                "expected_facts": {
                    "new_customers": _as_int(customer_row[0]),
                    "returning_customers": _as_int(customer_row[1]),
                    "unique_customers": unique_customers,
                    "returning_share_pct": returning_share,
                },
                "expected_evidence": {
                    "query_key": "customer_new_vs_returning_month",
                    "params": {
                        "merchant_id": merchant_id,
                        "month_start": month_start.isoformat(),
                        "month_end": config.end_date.isoformat(),
                    },
                },
                "pass_fail_criteria": {
                    "exact_integer_match": True,
                    "numeric_tolerance": 0.02,
                },
            }
        )

        # 7) Top customers by income this month
        top_customers = conn.execute(
            """
            SELECT c.customer_id, c.customer_display_name, SUM(t.amount) AS income
            FROM transactions t
            JOIN customers c ON t.customer_id = c.customer_id AND t.merchant_id = c.merchant_id
            WHERE t.merchant_id = ?
              AND CAST(t.occurred_at AS DATE) BETWEEN ? AND ?
            GROUP BY c.customer_id, c.customer_display_name
            ORDER BY income DESC, c.customer_id ASC
            LIMIT 3
            """,
            [merchant_id, month_start.isoformat(), config.end_date.isoformat()],
        ).fetchall()
        items.append(
            {
                "eval_id": "E07",
                "profile_id": _meta_for(question_catalog, "E07")[0],
                "persona_id": _meta_for(question_catalog, "E07")[0],
                "merchant_id": merchant_id,
                "question_es": _meta_for(question_catalog, "E07")[1],
                "expected_intent_id": "top_customers",
                "chart_allowed": True,
                "proactive_expected": False,
                "expected_facts": {
                    "top_customers": [
                        {
                            "customer_id": str(row[0]),
                            "customer_display_name": str(row[1]),
                            "income": _money(row[2]),
                        }
                        for row in top_customers
                    ]
                },
                "expected_evidence": {
                    "query_key": "top_customers_income_month_top3",
                    "params": {
                        "merchant_id": merchant_id,
                        "month_start": month_start.isoformat(),
                        "month_end": config.end_date.isoformat(),
                    },
                },
                "pass_fail_criteria": {
                    "top_order_must_match": True,
                    "numeric_tolerance": 0.01,
                },
            }
        )

        # 8) Inactive customers in last 30 days
        inactive_count = _query_one(
            conn,
            """
            WITH historical AS (
              SELECT DISTINCT customer_id
              FROM transactions
              WHERE merchant_id = ?
                AND CAST(occurred_at AS DATE) < ?
            ), recent AS (
              SELECT DISTINCT customer_id
              FROM transactions
              WHERE merchant_id = ?
                AND CAST(occurred_at AS DATE) BETWEEN ? AND ?
            )
            SELECT COUNT(*)
            FROM historical h
            LEFT JOIN recent r USING (customer_id)
            WHERE r.customer_id IS NULL
            """,
            [
                merchant_id,
                (config.end_date - timedelta(days=29)).isoformat(),
                merchant_id,
                (config.end_date - timedelta(days=29)).isoformat(),
                config.end_date.isoformat(),
            ],
        )[0]
        items.append(
            {
                "eval_id": "E08",
                "profile_id": _meta_for(question_catalog, "E08")[0],
                "persona_id": _meta_for(question_catalog, "E08")[0],
                "merchant_id": merchant_id,
                "question_es": _meta_for(question_catalog, "E08")[1],
                "expected_intent_id": "inactive_customers",
                "chart_allowed": False,
                "proactive_expected": False,
                "expected_facts": {"inactive_customers_30d": _as_int(inactive_count)},
                "expected_evidence": {
                    "query_key": "inactive_customers_30d",
                    "params": {
                        "merchant_id": merchant_id,
                        "window_days": 30,
                        "end_date": config.end_date.isoformat(),
                    },
                },
                "pass_fail_criteria": {
                    "exact_integer_match": True,
                },
            }
        )

        # 9) Peak hour this week
        peak_hour_row = _query_one(
            conn,
            """
            SELECT EXTRACT('hour' FROM CAST(occurred_at AS TIMESTAMP)) AS h,
                   SUM(amount) AS income
            FROM transactions
            WHERE merchant_id = ?
              AND CAST(occurred_at AS DATE) BETWEEN ? AND ?
            GROUP BY h
            ORDER BY income DESC, h ASC
            LIMIT 1
            """,
            [merchant_id, week_start.isoformat(), week_end.isoformat()],
        )
        items.append(
            {
                "eval_id": "E09",
                "profile_id": _meta_for(question_catalog, "E09")[0],
                "persona_id": _meta_for(question_catalog, "E09")[0],
                "merchant_id": merchant_id,
                "question_es": _meta_for(question_catalog, "E09")[1],
                "expected_intent_id": "peak_hours",
                "chart_allowed": True,
                "proactive_expected": False,
                "expected_facts": {
                    "peak_hour": _as_int(peak_hour_row[0]),
                    "peak_hour_income": _money(_as_float(peak_hour_row[1])),
                    "currency": "USD",
                },
                "expected_evidence": {
                    "query_key": "peak_hour_income_week",
                    "params": {
                        "merchant_id": merchant_id,
                        "start_date": week_start.isoformat(),
                        "end_date": week_end.isoformat(),
                    },
                },
                "pass_fail_criteria": {
                    "exact_integer_match": True,
                    "numeric_tolerance": 0.01,
                },
            }
        )

        # 10) Seller performance this week
        seller_perf = conn.execute(
            """
            SELECT s.seller_id, s.seller_display_name, s.role, SUM(t.amount) AS income, COUNT(*) AS tx_count
            FROM transactions t
            JOIN sellers s ON t.seller_id = s.seller_id AND t.merchant_id = s.merchant_id
            WHERE t.merchant_id = ?
              AND CAST(t.occurred_at AS DATE) BETWEEN ? AND ?
            GROUP BY s.seller_id, s.seller_display_name, s.role
            ORDER BY income DESC, s.seller_id ASC
            """,
            [merchant_id, week_start.isoformat(), week_end.isoformat()],
        ).fetchall()
        items.append(
            {
                "eval_id": "E10",
                "profile_id": _meta_for(question_catalog, "E10")[0],
                "persona_id": _meta_for(question_catalog, "E10")[0],
                "merchant_id": merchant_id,
                "question_es": _meta_for(question_catalog, "E10")[1],
                "expected_intent_id": "seller_performance",
                "chart_allowed": True,
                "proactive_expected": False,
                "expected_facts": {
                    "seller_breakdown": [
                        {
                            "seller_id": str(row[0]),
                            "seller_display_name": str(row[1]),
                            "role": str(row[2]),
                            "income": _money(row[3]),
                            "tx_count": int(row[4]),
                        }
                        for row in seller_perf
                    ]
                },
                "expected_evidence": {
                    "query_key": "seller_performance_week",
                    "params": {
                        "merchant_id": merchant_id,
                        "start_date": week_start.isoformat(),
                        "end_date": week_end.isoformat(),
                    },
                },
                "pass_fail_criteria": {
                    "top_order_must_match": True,
                    "numeric_tolerance": 0.01,
                },
            }
        )

        # 11) Last 7 days trend points
        trend_rows = conn.execute(
            """
            SELECT CAST(occurred_at AS DATE) AS d, SUM(amount) AS income
            FROM transactions
            WHERE merchant_id = ?
              AND CAST(occurred_at AS DATE) BETWEEN ? AND ?
            GROUP BY d
            ORDER BY d ASC
            """,
            [merchant_id, week_start.isoformat(), week_end.isoformat()],
        ).fetchall()
        items.append(
            {
                "eval_id": "E11",
                "profile_id": _meta_for(question_catalog, "E11")[0],
                "persona_id": _meta_for(question_catalog, "E11")[0],
                "merchant_id": merchant_id,
                "question_es": _meta_for(question_catalog, "E11")[1],
                "expected_intent_id": "income_period",
                "chart_allowed": True,
                "proactive_expected": False,
                "expected_facts": {
                    "daily_income": [
                        {"date": str(row[0]), "income": _money(row[1])} for row in trend_rows
                    ],
                    "currency": "USD",
                },
                "expected_evidence": {
                    "query_key": "daily_income_week",
                    "params": {
                        "merchant_id": merchant_id,
                        "start_date": week_start.isoformat(),
                        "end_date": week_end.isoformat(),
                    },
                },
                "pass_fail_criteria": {
                    "series_length_min": 5,
                    "numeric_tolerance": 0.01,
                },
            }
        )

        # 12) Top customer concentration this month
        concentration_row = _query_one(
            conn,
            """
            WITH ci AS (
              SELECT customer_id, SUM(amount) AS income
              FROM transactions
              WHERE merchant_id = ?
                AND CAST(occurred_at AS DATE) BETWEEN ? AND ?
              GROUP BY customer_id
            ), ranked AS (
              SELECT income,
                     ROW_NUMBER() OVER (ORDER BY income DESC) AS rn,
                     SUM(income) OVER () AS total_income
              FROM ci
            )
            SELECT
              COALESCE(SUM(CASE WHEN rn <= 1 THEN income END) / NULLIF(MAX(total_income),0), 0) * 100 AS top1_share,
              COALESCE(SUM(CASE WHEN rn <= 5 THEN income END) / NULLIF(MAX(total_income),0), 0) * 100 AS top5_share
            FROM ranked
            """,
            [merchant_id, month_start.isoformat(), config.end_date.isoformat()],
        )
        items.append(
            {
                "eval_id": "E12",
                "profile_id": _meta_for(question_catalog, "E12")[0],
                "persona_id": _meta_for(question_catalog, "E12")[0],
                "merchant_id": merchant_id,
                "question_es": _meta_for(question_catalog, "E12")[1],
                "expected_intent_id": "top_customers",
                "chart_allowed": False,
                "proactive_expected": bool(_as_float(concentration_row[0]) >= 20.0),
                "expected_facts": {
                    "top1_share_pct": round(_as_float(concentration_row[0]), 2),
                    "top5_share_pct": round(_as_float(concentration_row[1]), 2),
                },
                "expected_evidence": {
                    "query_key": "customer_concentration_month",
                    "params": {
                        "merchant_id": merchant_id,
                        "month_start": month_start.isoformat(),
                        "month_end": config.end_date.isoformat(),
                    },
                },
                "pass_fail_criteria": {
                    "numeric_tolerance": 0.02,
                    "must_not_hallucinate": True,
                },
            }
        )

        # 13) Income share by ticket-size buckets this week
        items.append(
            {
                "eval_id": "E13",
                "profile_id": _meta_for(question_catalog, "E13")[0],
                "persona_id": _meta_for(question_catalog, "E13")[0],
                "merchant_id": merchant_id,
                "question_es": _meta_for(question_catalog, "E13")[1],
                "expected_intent_id": "income_distribution",
                "chart_allowed": False,
                "proactive_expected": False,
                "expected_facts": {
                    "period": {"start": week_start.isoformat(), "end": week_end.isoformat()},
                    "small_income_share_pct": small_income_share_pct,
                    "medium_income_share_pct": medium_income_share_pct,
                    "large_income_share_pct": large_income_share_pct,
                    "currency": "USD",
                },
                "expected_evidence": {
                    "query_key": "income_share_by_ticket_size_week",
                    "params": {
                        "merchant_id": merchant_id,
                        "start_date": week_start.isoformat(),
                        "end_date": week_end.isoformat(),
                    },
                },
                "pass_fail_criteria": {
                    "numeric_tolerance": 0.02,
                    "must_not_hallucinate": True,
                },
            }
        )

        # 14) Ticket-size transaction counts this week
        items.append(
            {
                "eval_id": "E14",
                "profile_id": _meta_for(question_catalog, "E14")[0],
                "persona_id": _meta_for(question_catalog, "E14")[0],
                "merchant_id": merchant_id,
                "question_es": _meta_for(question_catalog, "E14")[1],
                "expected_intent_id": "income_distribution",
                "chart_allowed": False,
                "proactive_expected": False,
                "expected_facts": {
                    "period": {"start": week_start.isoformat(), "end": week_end.isoformat()},
                    "tx_count": tx_count_week,
                    "small_tx_count": small_tx_count,
                    "medium_tx_count": medium_tx_count,
                    "large_tx_count": large_tx_count,
                },
                "expected_evidence": {
                    "query_key": "tx_count_by_ticket_size_week",
                    "params": {
                        "merchant_id": merchant_id,
                        "start_date": week_start.isoformat(),
                        "end_date": week_end.isoformat(),
                    },
                },
                "pass_fail_criteria": {
                    "exact_integer_match": True,
                    "must_not_hallucinate": True,
                },
            }
        )

        # 15) Explicit proactive check question
        items.append(
            {
                "eval_id": "E15",
                "profile_id": _meta_for(question_catalog, "E15")[0],
                "persona_id": _meta_for(question_catalog, "E15")[0],
                "merchant_id": merchant_id,
                "question_es": _meta_for(question_catalog, "E15")[1],
                "expected_intent_id": "income_vs_previous",
                "chart_allowed": False,
                "proactive_expected": bool(pct_change is not None and pct_change <= -10.0),
                "expected_facts": {
                    "current_period_income": _money(_as_float(compare_row[0])),
                    "previous_period_income": _money(_as_float(compare_row[1])),
                    "pct_change": pct_change,
                    "should_include_guidance": True,
                },
                "expected_evidence": {
                    "query_key": "income_vs_previous_week",
                    "params": {
                        "merchant_id": merchant_id,
                        "current_start": week_start.isoformat(),
                        "current_end": week_end.isoformat(),
                        "previous_start": prev_week_start.isoformat(),
                        "previous_end": prev_week_end.isoformat(),
                    },
                },
                "pass_fail_criteria": {
                    "numeric_tolerance": 0.02,
                    "must_include_followup_guidance": True,
                    "must_not_hallucinate": True,
                },
            }
        )

        # 16) Unsupported intent: inventory
        items.append(
            {
                "eval_id": "E16",
                "profile_id": _meta_for(question_catalog, "E16")[0],
                "persona_id": _meta_for(question_catalog, "E16")[0],
                "merchant_id": merchant_id,
                "question_es": _meta_for(question_catalog, "E16")[1],
                "expected_intent_id": "unsupported",
                "chart_allowed": False,
                "proactive_expected": False,
                "expected_facts": {
                    "must_refuse": True,
                    "reason": "inventory_not_in_dataset",
                },
                "expected_evidence": {
                    "policy_key": "unsupported_topic_inventory",
                },
                "pass_fail_criteria": {
                    "must_refuse_clearly": True,
                    "must_offer_supported_alternative": True,
                    "must_not_hallucinate": True,
                },
            }
        )

        # 17) Unsupported intent: profit
        items.append(
            {
                "eval_id": "E17",
                "profile_id": _meta_for(question_catalog, "E17")[0],
                "persona_id": _meta_for(question_catalog, "E17")[0],
                "merchant_id": merchant_id,
                "question_es": _meta_for(question_catalog, "E17")[1],
                "expected_intent_id": "unsupported",
                "chart_allowed": False,
                "proactive_expected": False,
                "expected_facts": {
                    "must_refuse": True,
                    "reason": "profit_not_in_dataset",
                },
                "expected_evidence": {
                    "policy_key": "unsupported_topic_profit",
                },
                "pass_fail_criteria": {
                    "must_refuse_clearly": True,
                    "must_offer_supported_alternative": True,
                    "must_not_hallucinate": True,
                },
            }
        )

    return items


def build_eval_set(config: EvalBuildConfig) -> dict[str, object]:
    """Build and persist the evaluation set artifact.

    Args:
        config: Evaluation build configuration.

    Returns:
        dict[str, object]: Full eval-set payload.

    Raises:
        OSError: If output file cannot be written.
        duckdb.Error: If SQL execution fails.
    """
    items = _build_items(config)
    payload = {
        "version": "v1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "merchant_id": config.merchant_id,
        "reference_end_date": config.end_date.isoformat(),
        "item_count": len(items),
        "items": items,
    }

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    return payload


def main() -> None:
    """CLI entrypoint to build the initial MVP evaluation set.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit: If CLI parsing fails.
    """
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build deterministic evaluation set for DeUna MVP")
    parser.add_argument(
        "--merchant-id",
        type=str,
        default=None,
        help="Optional merchant override (falls back to EVAL_MERCHANT_ID or DEUNA_MERCHANT_ID).",
    )
    args = parser.parse_args()

    config = _load_config(args.merchant_id)
    payload = build_eval_set(config)

    print("Evaluation set generated successfully.")
    print(f"Output file: {config.output_path}")
    print(f"Items: {payload['item_count']}")


if __name__ == "__main__":
    main()
