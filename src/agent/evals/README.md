# Evals Package Overview

## Purpose
This package generates and stores a deterministic MVP evaluation set grounded on the synthetic dataset.
The goal is to measure correctness without relying on subjective manual checking.

## What it creates
The builder writes one JSON artifact:
- data/evals/eval_set_v1.json

Each evaluation item includes:
- question text in Spanish
- expected intent
- expected facts (numbers/dates/categories)
- expected evidence metadata
- chart policy (allowed or not)
- proactive guidance expectation
- pass/fail criteria

## Current scope
The current builder generates 15 items for one merchant at a time.
Merchant selection follows this order:
1. CLI flag `--merchant-id`
2. `EVAL_MERCHANT_ID`
3. `DEUNA_MERCHANT_ID`

Question wording and profile assignment come from:
- `src/agent/evals/profiles_question_bank.yaml`

This means you can update profile style and question text without changing SQL logic.

## Why deterministic
Expected values are computed directly from DuckDB using fixed SQL queries.
This avoids brittle hard-coded values and keeps evals synchronized with regenerated datasets.

## How to run
From repository root:
- `uv run evals-build`
- `uv run evals-build --merchant-id M002`

## Design note
This package focuses on evaluation artifact creation only.
Execution/scoring against live model responses will be added in a later milestone.

## Current question set (15)
These are the current Spanish question prompts generated for `eval_set_v1`:

1. ¿Cuánto gané esta semana?
2. ¿Cuánto vendí esta semana?
3. ¿Cuánto gané ayer?
4. ¿Qué días me fue mejor y peor?
5. ¿Cómo me fue respecto a la semana pasada?
6. ¿Cuántos clientes de este mes fueron nuevos y cuántos regresaron?
7. ¿Cuáles son mis mejores tres clientes?
8. ¿Cuántos clientes no volvieron en el último mes?
9. ¿En qué hora vendo más?
10. ¿Qué tal les fue a mis vendedores esta semana?
11. ¿Cuánto gané por día esta semana?
12. ¿Estoy dependiendo de pocos clientes este mes?
13. ¿Cuánto inventario me queda?
14. ¿Cuál fue mi ganancia neta esta semana?
15. Dime si esta semana voy mejor o peor que la anterior y qué debería revisar primero.

Notes:
- Questions 13 and 14 are intentionally out-of-scope with current data.
- They stay in the eval set to test safe refusal and helpful redirection behavior.
