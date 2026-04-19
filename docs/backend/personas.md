# Perfiles para DeUna MVP

La fuente oficial de perfiles se mantiene en:
- `src/agent/semantics/profiles.yaml`

Este documento queda como referencia breve para lectura humana.

## Integración con evaluación
- Preguntas seleccionadas para evaluación SQL-grounded: `src/agent/semantics/eval_question_bank.yaml`.
- Refinamiento de preguntas por perfil con LLM: ocurre dentro de `uv run evals-build`.
