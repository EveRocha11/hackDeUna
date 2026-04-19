"""Run automatic conversational agent evaluations over eval_set_v1."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage
from langsmith import Client
from langsmith.evaluation import RunEvaluator, run_evaluator
from langsmith.schemas import Example, Run
from openevals.llm import create_llm_as_judge
from openevals.trajectory import create_trajectory_match_evaluator

from server.langgraph_agent.graph import graph


def _load_eval_items(path: Path) -> list[dict[str, Any]]:
    """Load evaluation items from eval set artifact.

    Args:
        path: Eval JSON path.

    Returns:
        list[dict[str, Any]]: Eval item list.

    Raises:
        FileNotFoundError: If eval file does not exist.
        ValueError: If JSON structure is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Eval set file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Eval set payload must be an object")

    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("Eval set must contain an 'items' list")
    return items


def _extract_messages(run_output: dict[str, Any]) -> list[BaseMessage]:
    """Extract message trajectory from graph run output.

    Args:
        run_output: Graph invocation output.

    Returns:
        list[BaseMessage]: Message list.

    Raises:
        ValueError: If output does not include messages.
    """
    messages = run_output.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Graph output missing messages list")
    return messages


def _messages_to_trajectory(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert LangChain message objects into OpenAI-style trajectory dicts.

    Args:
        messages: LangChain messages returned by graph invocation.

    Returns:
        list[dict[str, Any]]: JSON-serializable trajectory payload.

    Raises:
        None.
    """
    trajectory: list[dict[str, Any]] = []
    for msg in messages:
        if msg.type == "human":
            content = str(msg.content) if not isinstance(msg.content, str) else msg.content
            trajectory.append({"role": "user", "content": content})
            continue
        if msg.type == "tool":
            content = str(msg.content) if not isinstance(msg.content, str) else msg.content
            tool_call_id = getattr(msg, "tool_call_id", "")
            trajectory.append(
                {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": str(tool_call_id),
                }
            )
            continue

        if isinstance(msg, AIMessage):
            content = str(msg.content) if not isinstance(msg.content, str) else msg.content
            tool_calls: list[dict[str, Any]] = []
            for idx, call in enumerate(msg.tool_calls, start=1):
                tool_calls.append(
                    {
                        "id": str(call.get("id") or f"call_{idx}"),
                        "type": "function",
                        "function": {
                            "name": str(call.get("name", "")),
                            "arguments": json.dumps(call.get("args", {}), ensure_ascii=False),
                        },
                    }
                )

            item: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                item["tool_calls"] = tool_calls
            trajectory.append(item)
            continue

        content = str(msg.content) if not isinstance(msg.content, str) else msg.content
        trajectory.append({"role": "assistant", "content": content})

    return trajectory


def _extract_provenance_from_trajectory(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract query provenance payloads from tool trajectory messages.

    Args:
        trajectory: OpenAI-style message trajectory.

    Returns:
        list[dict[str, Any]]: Parsed provenance objects.

    Raises:
        None.
    """
    provenance: list[dict[str, Any]] = []
    for item in trajectory:
        if not isinstance(item, dict):
            continue
        if str(item.get("role", "")) != "tool":
            continue
        content = item.get("content")
        if not isinstance(content, str):
            continue
        try:
            parsed = json.loads(content)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        if "query_key" not in parsed and "resolved_sql" not in parsed:
            continue
        provenance.append(parsed)
    return provenance


def _provenance_passed(
    *,
    expected_intent_id: str,
    provenance_items: list[dict[str, Any]],
) -> bool:
    """Validate provenance coverage for one eval case.

    Args:
        expected_intent_id: Expected intent id from eval item.
        provenance_items: Parsed provenance payloads.

    Returns:
        bool: Whether provenance requirements are satisfied.

    Raises:
        None.
    """
    if expected_intent_id == "unsupported":
        return True
    if not provenance_items:
        return False

    for item in provenance_items:
        if not bool(item.get("ok", False)):
            continue
        if str(item.get("intent_id", "")).strip() != expected_intent_id:
            continue
        if not str(item.get("query_key", "")).strip():
            continue
        if not str(item.get("resolved_sql", "")).strip():
            continue
        return True
    return False


def _final_ai_content(messages: list[BaseMessage]) -> str:
    """Return final AI text content from a message trajectory.

    Args:
        messages: LangChain message trajectory.

    Returns:
        str: Final AI message content.

    Raises:
        None.
    """
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            return msg.content.strip()
    return ""


def _expected_tool_names(eval_item: dict[str, Any]) -> list[str]:
    """Map eval item to required tool usage expectations.

    Args:
        eval_item: Evaluation item payload.

    Returns:
        list[str]: Required tool names for trajectory checks.

    Raises:
        None.
    """
    expected_intent = str(eval_item.get("expected_intent_id", "")).strip()
    if expected_intent == "unsupported":
        return []
    return ["run_analytics_query_tool"]


def _trajectory_tool_names(trajectory_payload: list[dict[str, Any]]) -> list[str]:
    """Extract ordered tool names from an OpenAI-style trajectory payload.

    Args:
        trajectory_payload: Message trajectory payload.

    Returns:
        list[str]: Tool names as they were called.

    Raises:
        None.
    """
    tool_names: list[str] = []
    for message in trajectory_payload:
        if str(message.get("role", "")).strip() != "assistant":
            continue
        tool_calls = message.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", {})
            if not isinstance(function, dict):
                continue
            name = str(function.get("name", "")).strip()
            if name:
                tool_names.append(name)
    return tool_names


def _reference_trajectory(question_es: str, tool_names: list[str]) -> list[dict[str, Any]]:
    """Build minimal reference trajectory for tool-usage scoring.

    Args:
        question_es: User question text.
        tool_names: Required tool names.

    Returns:
        list[dict[str, Any]]: OpenAI-style reference trajectory.

    Raises:
        None.
    """
    trajectory: list[dict[str, Any]] = [{"role": "user", "content": question_es}]
    if not tool_names:
        trajectory.append({"role": "assistant", "content": ""})
        return trajectory

    tool_calls = [
        {
            "id": f"ref_call_{idx}",
            "type": "function",
            "function": {"name": name, "arguments": "{}"},
        }
        for idx, name in enumerate(tool_names, start=1)
    ]
    trajectory.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
    return trajectory


def _evaluator_result_to_dict(result: Any) -> dict[str, Any]:
    """Normalize evaluator outputs into a dict payload.

    Args:
        result: Evaluator return value.

    Returns:
        dict[str, Any]: Normalized dictionary payload.

    Raises:
        None.
    """
    if isinstance(result, dict):
        return dict(result)
    if isinstance(result, list):
        items: list[dict[str, Any]] = []
        for item in result:
            if isinstance(item, dict):
                items.append(dict(item))
            else:
                items.append({"value": str(item)})
        return {"results": items}
    return {"value": result}


def _evaluator_passed(result: Any) -> bool:
    """Compute pass/fail from evaluator return shape.

    Args:
        result: Evaluator return value.

    Returns:
        bool: Whether evaluation passed.

    Raises:
        None.
    """
    if isinstance(result, dict):
        return bool(result.get("score", False))
    if isinstance(result, list):
        dict_items = [item for item in result if isinstance(item, dict)]
        if not dict_items:
            return False
        return all(bool(item.get("score", False)) for item in dict_items)
    return False


def _normalize_eval_item(eval_item: dict[str, Any]) -> dict[str, Any]:
    """Normalize one eval item to safe JSON payload for dataset outputs.

    Args:
        eval_item: Raw eval item from eval set JSON.

    Returns:
        dict[str, Any]: Normalized output payload.

    Raises:
        None.
    """
    return {
        "eval_id": str(eval_item.get("eval_id", "")).strip(),
        "expected_intent_id": str(eval_item.get("expected_intent_id", "")).strip(),
        "expected_facts": eval_item.get("expected_facts", {}),
        "pass_fail_criteria": eval_item.get("pass_fail_criteria", {}),
    }


def ensure_langsmith_dataset(
    client: Client,
    dataset_name: str,
    eval_items: list[dict[str, Any]],
    update: bool,
) -> None:
    """Create or update LangSmith dataset from eval_set items.

    Args:
        client: LangSmith client instance.
        dataset_name: Target dataset name.
        eval_items: Eval items loaded from eval_set JSON.
        update: Whether to update existing examples.

    Returns:
        None.

    Raises:
        None.
    """
    examples_payload = [
        {
            "inputs": {
                "question_es": str(item.get("question_es", "")).strip(),
            },
            "outputs": _normalize_eval_item(item),
        }
        for item in eval_items
        if str(item.get("question_es", "")).strip()
    ]

    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
        if not update:
            return

        existing_by_question: dict[str, Example] = {}
        for existing in client.list_examples(dataset_id=dataset.id):
            inputs = existing.inputs or {}
            question_es = str(inputs.get("question_es", "")).strip()
            if question_es:
                existing_by_question[question_es] = existing

        updates: list[dict[str, Any]] = []
        new_examples: list[dict[str, Any]] = []
        for candidate in examples_payload:
            question_es = str(candidate["inputs"]["question_es"]).strip()
            previous = existing_by_question.get(question_es)
            if previous is None:
                new_examples.append(candidate)
                continue
            if (previous.outputs or {}) != candidate["outputs"]:
                updates.append(
                    {
                        "id": previous.id,
                        "inputs": candidate["inputs"],
                        "outputs": candidate["outputs"],
                    }
                )

        if updates:
            client.update_examples(dataset_id=dataset.id, updates=updates)
        if new_examples:
            client.create_examples(dataset_id=dataset.id, examples=new_examples)
        return

    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(dataset_id=dataset.id, examples=examples_payload)


def _run_agent_for_langsmith(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run graph for one LangSmith dataset input.

    Args:
        inputs: Dataset example inputs with `question_es`.

    Returns:
        dict[str, Any]: Structured run outputs used by evaluators.

    Raises:
        KeyError: If question_es is missing.
    """
    question_es = str(inputs["question_es"]).strip()
    thread_id = f"langsmith-{datetime.now(timezone.utc).timestamp()}"
    run_output = graph.invoke(
        {"messages": [{"role": "user", "content": question_es}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    messages = _extract_messages(run_output)
    answer_es = _final_ai_content(messages)

    tool_call_names: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls:
                name = call.get("name")
                if isinstance(name, str) and name:
                    tool_call_names.append(name)

    return {
        "answer_es": answer_es,
        "tool_call_names": tool_call_names,
        "trajectory": _messages_to_trajectory(messages),
    }


def _build_langsmith_evaluators() -> list[RunEvaluator]:
    """Build LangSmith run evaluators for trajectory and quality.

    Args:
        None.

    Returns:
        list[RunEvaluator]: Configured evaluators.

    Raises:
        None.
    """
    trajectory_superset_eval = create_trajectory_match_evaluator(
        trajectory_match_mode="superset",
        tool_args_match_mode="ignore",
    )
    trajectory_subset_eval = create_trajectory_match_evaluator(
        trajectory_match_mode="subset",
        tool_args_match_mode="ignore",
    )
    quality_eval = create_llm_as_judge(
        prompt=(
            "Evalua si la respuesta del asistente es correcta y util para la pregunta.\n"
            "Question: {inputs}\n"
            "Assistant output: {outputs}\n"
            "Expected reference facts and intent: {reference_outputs}\n"
            "Responde con true solo si es correcta, relevante y segura."
        ),
        model="openai:gpt-4.1-mini",
        feedback_key="quality_score",
    )

    def _trajectory(run: Run, example: Example | None) -> dict[str, Any]:
        run_outputs = run.outputs or {}
        trajectory = run_outputs.get("trajectory", [])
        if not isinstance(trajectory, list):
            trajectory = []

        example_inputs = {} if example is None else (example.inputs or {})
        example_outputs = {} if example is None else (example.outputs or {})
        question_es = str(example_inputs.get("question_es", "")).strip()
        expected_intent = str(example_outputs.get("expected_intent_id", "")).strip()
        if expected_intent == "unsupported":
            actual_tools = _trajectory_tool_names(trajectory)
            allowed = {"run_analytics_query_tool", "assistant_capabilities_tool"}
            score = all(tool_name in allowed for tool_name in actual_tools)
            return {
                "key": "trajectory_unsupported_tools",
                "score": score,
                "comment": None if score else f"unsupported_tool_calls_detected: {actual_tools}",
            }

        expected_tools = [] if expected_intent == "unsupported" else ["run_analytics_query_tool"]
        reference = _reference_trajectory(question_es, expected_tools)

        result = (
            trajectory_superset_eval(outputs=trajectory, reference_outputs=reference)
            if expected_tools
            else trajectory_subset_eval(outputs=trajectory, reference_outputs=reference)
        )
        return _evaluator_result_to_dict(result)

    def _quality(run: Run, example: Example | None) -> dict[str, Any]:
        example_inputs = {} if example is None else (example.inputs or {})
        example_outputs = {} if example is None else (example.outputs or {})
        run_outputs = run.outputs or {}
        answer_es = str(run_outputs.get("answer_es", ""))

        try:
            result = quality_eval(
                inputs=str(example_inputs.get("question_es", "")),
                outputs=answer_es,
                reference_outputs=json.dumps(example_outputs, ensure_ascii=False),
            )
        except Exception as exc:
            result = {
                "key": "quality_score",
                "score": False,
                "comment": f"quality_eval_error: {exc}",
            }
        return _evaluator_result_to_dict(result)

    def _provenance(run: Run, example: Example | None) -> dict[str, Any]:
        run_outputs = run.outputs or {}
        trajectory = run_outputs.get("trajectory", [])
        if not isinstance(trajectory, list):
            trajectory = []

        example_outputs = {} if example is None else (example.outputs or {})
        expected_intent = str(example_outputs.get("expected_intent_id", "")).strip()
        provenance_items = _extract_provenance_from_trajectory(trajectory)
        score = _provenance_passed(
            expected_intent_id=expected_intent,
            provenance_items=provenance_items,
        )
        return {
            "key": "provenance_score",
            "score": score,
            "comment": None if score else "missing_or_invalid_query_provenance",
        }

    return [run_evaluator(_trajectory), run_evaluator(_quality), run_evaluator(_provenance)]


def _langsmith_results_to_json(results: Any) -> object:
    """Convert LangSmith evaluate return payload to JSON-safe object.

    Args:
        results: Result returned by Client.evaluate.

    Returns:
        object: JSON-serializable payload.

    Raises:
        None.
    """
    if hasattr(results, "_results"):
        try:
            return json.loads(json.dumps(results._results, default=str))
        except Exception:
            return str(results._results)
    return str(results)


def run_langsmith_eval(
    *,
    eval_set_path: Path,
    dataset_name: str,
    experiment_prefix: str,
    update_dataset: bool,
    max_concurrency: int,
    report_path: Path,
) -> dict[str, Any]:
    """Run baseline evaluation in LangSmith with dataset + evaluators.

    Args:
        eval_set_path: Source eval set JSON file.
        dataset_name: LangSmith dataset name.
        experiment_prefix: LangSmith experiment prefix.
        update_dataset: Whether to update examples if dataset already exists.
        max_concurrency: Maximum evaluation concurrency.
        report_path: Local JSON artifact path for exported results.

    Returns:
        dict[str, Any]: Minimal run metadata with experiment URL if available.

    Raises:
        ValueError: If required LangSmith settings are missing.
    """
    if not os.getenv("LANGSMITH_API_KEY"):
        raise ValueError("LANGSMITH_API_KEY is required for LangSmith evaluation")

    items = _load_eval_items(eval_set_path)
    client = Client()
    ensure_langsmith_dataset(client, dataset_name, items, update_dataset)

    experiment_results = client.evaluate(
        _run_agent_for_langsmith,
        data=dataset_name,
        evaluators=_build_langsmith_evaluators(),
        experiment_prefix=experiment_prefix,
        max_concurrency=max(1, max_concurrency),
    )

    report_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_set_path": str(eval_set_path),
        "dataset_name": dataset_name,
        "experiment_prefix": experiment_prefix,
        "experiment_url": str(getattr(experiment_results, "url", "")),
        "results": _langsmith_results_to_json(experiment_results),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "dataset_name": dataset_name,
        "experiment_url": str(getattr(experiment_results, "url", "")),
        "report_path": str(report_path),
    }


def run_agent_eval(
    *,
    eval_set_path: Path,
    report_path: Path,
    limit: int | None,
) -> dict[str, Any]:
    """Execute automatic evals for conversational agent quality and trajectories.

    Args:
        eval_set_path: Input eval set JSON path.
        report_path: Output report JSON path.
        limit: Optional max number of eval items.

    Returns:
        dict[str, Any]: Aggregated evaluation report.

    Raises:
        ValueError: If evaluator initialization fails.
    """
    items = _load_eval_items(eval_set_path)
    if limit is not None:
        items = items[: max(0, limit)]

    trajectory_superset_eval = create_trajectory_match_evaluator(
        trajectory_match_mode="superset",
        tool_args_match_mode="ignore",
    )
    trajectory_subset_eval = create_trajectory_match_evaluator(
        trajectory_match_mode="subset",
        tool_args_match_mode="ignore",
    )

    quality_eval = create_llm_as_judge(
        prompt=(
            "Evalua si la respuesta del asistente es correcta y util para la pregunta.\n"
            "Question: {inputs}\n"
            "Assistant output: {outputs}\n"
            "Expected reference facts and intent: {reference_outputs}\n"
            "Responde con true solo si es correcta, relevante y segura."
        ),
        model="openai:gpt-4.1-mini",
        feedback_key="quality_score",
    )

    case_results: list[dict[str, Any]] = []
    trajectory_pass_count = 0
    quality_pass_count = 0
    provenance_pass_count = 0

    for idx, item in enumerate(items, start=1):
        question_es = str(item.get("question_es", "")).strip()
        eval_id = str(item.get("eval_id", f"item_{idx}")).strip()
        if not question_es:
            continue

        thread_id = f"eval-{eval_id}-{idx}"
        run_output = graph.invoke(
            {"messages": [{"role": "user", "content": question_es}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        messages = _extract_messages(run_output)
        answer_es = _final_ai_content(messages)
        trajectory_payload = _messages_to_trajectory(messages)
        provenance_items = _extract_provenance_from_trajectory(trajectory_payload)

        expected_tools = _expected_tool_names(item)
        expected_intent = str(item.get("expected_intent_id", "")).strip()
        if expected_intent == "unsupported":
            actual_tools = _trajectory_tool_names(trajectory_payload)
            allowed = {"run_analytics_query_tool", "assistant_capabilities_tool"}
            score = all(tool_name in allowed for tool_name in actual_tools)
            trajectory_result = {
                "key": "trajectory_unsupported_tools",
                "score": score,
                "comment": None if score else f"unsupported_tool_calls_detected: {actual_tools}",
            }
        else:
            reference_traj = _reference_trajectory(question_es, expected_tools)
            trajectory_result = trajectory_superset_eval(
                outputs=trajectory_payload,
                reference_outputs=reference_traj,
            )

        try:
            quality_result = quality_eval(
                inputs=question_es,
                outputs=answer_es,
                reference_outputs=json.dumps(
                    {
                        "expected_intent_id": item.get("expected_intent_id"),
                        "expected_facts": item.get("expected_facts"),
                        "pass_fail_criteria": item.get("pass_fail_criteria"),
                    },
                    ensure_ascii=False,
                ),
            )
        except Exception as exc:
            quality_result = {
                "key": "quality_score",
                "score": False,
                "comment": f"quality_eval_error: {exc}",
            }

        traj_score = _evaluator_passed(trajectory_result)
        qual_score = _evaluator_passed(quality_result)
        prov_score = _provenance_passed(
            expected_intent_id=str(item.get("expected_intent_id", "")).strip(),
            provenance_items=provenance_items,
        )
        trajectory_pass_count += 1 if traj_score else 0
        quality_pass_count += 1 if qual_score else 0
        provenance_pass_count += 1 if prov_score else 0

        case_results.append(
            {
                "eval_id": eval_id,
                "question_es": question_es,
                "expected_intent_id": item.get("expected_intent_id"),
                "answer_es": answer_es,
                "trajectory_eval": _evaluator_result_to_dict(trajectory_result),
                "quality_eval": _evaluator_result_to_dict(quality_result),
                "provenance_eval": {
                    "key": "provenance_score",
                    "score": prov_score,
                    "details": provenance_items,
                },
                "passed": traj_score and qual_score and prov_score,
            }
        )

    total = len(case_results)
    summary = {
        "total_cases": total,
        "trajectory_pass_count": trajectory_pass_count,
        "quality_pass_count": quality_pass_count,
        "provenance_pass_count": provenance_pass_count,
        "overall_pass_count": sum(1 for case in case_results if case["passed"]),
        "trajectory_pass_rate": 0.0 if total == 0 else round(trajectory_pass_count / total, 4),
        "quality_pass_rate": 0.0 if total == 0 else round(quality_pass_count / total, 4),
        "provenance_pass_rate": 0.0 if total == 0 else round(provenance_pass_count / total, 4),
        "overall_pass_rate": 0.0
        if total == 0
        else round(sum(1 for case in case_results if case["passed"]) / total, 4),
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_set_path": str(eval_set_path),
        "model": os.getenv("AGENT_MODEL", "gpt-4.1-mini"),
        "summary": summary,
        "cases": case_results,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    """CLI entrypoint for automatic conversational agent evaluation.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit: If CLI argument parsing fails.
    """
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run automatic agent evaluation over eval_set_v1")
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=Path("data/evals/eval_set_v1.json"),
        help="Path to eval set JSON artifact.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/evals/reports/agent_eval_report_v1.json"),
        help="Path for evaluation report JSON output.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of eval items to run.",
    )
    parser.add_argument(
        "--langsmith-dataset",
        type=str,
        default="",
        help="If set, run evaluation in LangSmith using this dataset name.",
    )
    parser.add_argument(
        "--langsmith-experiment-prefix",
        type=str,
        default="deuna-agent-baseline",
        help="Experiment prefix used by LangSmith evaluate.",
    )
    parser.add_argument(
        "--langsmith-max-concurrency",
        type=int,
        default=5,
        help="Max concurrency for LangSmith evaluation.",
    )
    parser.add_argument(
        "--langsmith-update-dataset",
        action="store_true",
        default=False,
        help="Update examples when dataset already exists.",
    )
    args = parser.parse_args()

    if args.langsmith_dataset:
        info = run_langsmith_eval(
            eval_set_path=args.eval_set,
            dataset_name=args.langsmith_dataset,
            experiment_prefix=args.langsmith_experiment_prefix,
            update_dataset=args.langsmith_update_dataset,
            max_concurrency=args.langsmith_max_concurrency,
            report_path=args.report,
        )
        print("LangSmith evaluation completed.")
        print(f"Dataset: {info['dataset_name']}")
        print(f"Experiment URL: {info['experiment_url']}")
        print(f"Report: {info['report_path']}")
        return

    report = run_agent_eval(
        eval_set_path=args.eval_set,
        report_path=args.report,
        limit=args.limit,
    )

    print("Agent evaluation completed.")
    print(f"Total cases: {report['summary']['total_cases']}")
    print(f"Trajectory pass rate: {report['summary']['trajectory_pass_rate']}")
    print(f"Quality pass rate: {report['summary']['quality_pass_rate']}")
    print(f"Provenance pass rate: {report['summary']['provenance_pass_rate']}")
    print(f"Overall pass rate: {report['summary']['overall_pass_rate']}")


if __name__ == "__main__":
    main()
