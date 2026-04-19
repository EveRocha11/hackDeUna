# hackDeUna

Backend MVP for DeUna assistant, with deterministic and conversational (LangGraph) API variants.

## Run The API Locally

```bash
uv sync --extra dev
make api-dev
```

The API runs at `http://localhost:8000`.

## Frontend Testing Endpoint (Simple)

Use this endpoint for frontend integration tests:

- Method: `POST`
- Path: `/api/v1/agent/query`
- Body:
	- `question` (string, required)
	- `thread_id` (string, optional)

Example request:

```bash
curl -X POST "http://localhost:8000/api/v1/agent/query" \
	-H "Content-Type: application/json" \
	-d '{
		"question": "¿Cuánto vendí esta semana?",
		"thread_id": "frontend-smoke-1"
	}'
```

Example response:

```json
{
	"answer": "Esta semana vendiste ...",
	"tools": ["run_analytics_query_tool"]
}
```

## Existing Agent Endpoint (Detailed)

The existing endpoint is still available:

- Method: `POST`
- Path: `/assistant/agent-query`
- Body:
	- `question_es` (string, required)
	- `thread_id` (string, optional)

Example request:

```bash
curl -X POST "http://localhost:8000/assistant/agent-query" \
	-H "Content-Type: application/json" \
	-d '{
		"question_es": "¿Cómo me fue esta semana en comparación con la anterior?",
		"thread_id": "agent-debug-1"
	}'
```

## CORS For Frontend Calls

The API now enables CORS and reads allowed origins from `API_CORS_ALLOW_ORIGINS`.

- Default: `*`
- Example restrictive value: `http://localhost:5173,http://localhost:3000`

Example:

```bash
API_CORS_ALLOW_ORIGINS="http://localhost:5173" make api-dev
```