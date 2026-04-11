from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openenv_support_triage.environment import SupportTriageEnv
from openenv_support_triage.graders import grade_state
from openenv_support_triage.models import ActionModel, ObservationModel
from openenv_support_triage.tasks import TASKS

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_API_BASE_URL = "https://api.openai.com/v1"
DEFAULT_SEED = 7
DEFAULT_MAX_RUNTIME_SECONDS = 20 * 60
LOG_EPS = 0.01
SCORE_EPS = 0.1


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _strict_log_reward(value: float) -> float:
    return min(1.0 - LOG_EPS, max(LOG_EPS, value))


def _strict_score(value: float) -> float:
    return min(1.0 - SCORE_EPS, max(SCORE_EPS, value))


def _one_decimal_score(value: float) -> float:
    return round(_strict_score(value), 1)


def _format_action(action: ActionModel) -> str:
    parts = [f"action_type={action.action_type}"]
    if action.ticket_id is not None:
        parts.append(f"ticket_id={action.ticket_id}")
    if action.priority is not None:
        parts.append(f"priority={action.priority}")
    if action.team is not None:
        parts.append(f"team={action.team}")
    if action.reply_text is not None:
        parts.append("reply_text=present")
    if action.resolution_note is not None:
        parts.append("resolution_note=present")
    return "|".join(parts)


def log_start(task_name: str, model_name: str) -> None:
    print(f"[START] task={task_name} env=openenv-support-triage model={model_name}", flush=True)


def log_step(step: int, action: ActionModel, reward: float, done: bool, error: str | None) -> None:
    error_value = error if error is not None else "null"
    reward = _strict_log_reward(reward)
    print(
        f"[STEP] step={step} action={_format_action(action)} reward={reward:.2f} "
        f"done={_bool_str(done)} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_text = ",".join(f"{_strict_log_reward(r):.2f}" for r in rewards)
    print(f"[END] success={_bool_str(success)} steps={steps} rewards={rewards_text}", flush=True)


def log_score(task_id: str, task_score: float, trajectory_reward: float) -> None:
    print(
        f"[SCORE] task={task_id} task_score={_one_decimal_score(task_score):.1f} "
        f"trajectory_reward={_one_decimal_score(trajectory_reward):.1f}",
        flush=True,
    )


def log_summary(aggregate_score: float, runtime_seconds: float) -> None:
    print(
        f"[SUMMARY] aggregate_score={_one_decimal_score(aggregate_score):.1f} runtime_seconds={runtime_seconds:.3f}",
        flush=True,
    )


def heuristic_action(observation: ObservationModel) -> ActionModel:
    for ticket in observation.tickets:
        if ticket.priority is None or ticket.team is None:
            text = f"{ticket.subject} {ticket.customer_message}".lower()
            if "fraud" in text or "unknown purchase" in text or "chargeback" in text:
                return ActionModel(action_type="classify_ticket", ticket_id=ticket.ticket_id, priority="urgent", team="risk")
            if "refund" in text or "invoice" in text or "prorated" in text or "charge" in text:
                priority = "high" if ticket.customer_tier in {"premium", "enterprise"} else "medium"
                return ActionModel(action_type="classify_ticket", ticket_id=ticket.ticket_id, priority=priority, team="billing")
            if "api" in text or "500" in text or "log in" in text or "password" in text:
                priority = "urgent" if "down" in text or "500" in text else "high"
                return ActionModel(action_type="classify_ticket", ticket_id=ticket.ticket_id, priority=priority, team="technical")
            return ActionModel(action_type="classify_ticket", ticket_id=ticket.ticket_id, priority="medium", team="support")

    for ticket in observation.tickets:
        if not ticket.drafted_reply and ticket.status != "resolved":
            reply = (
                "Thanks for contacting us. We will verify details, provide an update, "
                "and follow support policy."
            )
            return ActionModel(action_type="draft_reply", ticket_id=ticket.ticket_id, reply_text=reply)

    for ticket in observation.tickets:
        if ticket.status != "resolved":
            return ActionModel(
                action_type="resolve_ticket",
                ticket_id=ticket.ticket_id,
                resolution_note="Issue triaged, response drafted, and routed to correct team.",
            )

    return ActionModel(action_type="noop")


def llm_action(client: OpenAI, model: str, observation: ObservationModel, seed: int) -> ActionModel:
    schema_hint = {
        "action_type": "classify_ticket|draft_reply|resolve_ticket|noop",
        "ticket_id": "string or null",
        "priority": "low|medium|high|urgent or null",
        "team": "support|billing|technical|risk or null",
        "reply_text": "string or null",
        "resolution_note": "string or null",
    }

    prompt = {
        "objective": observation.objective,
        "step_index": observation.step_index,
        "max_steps": observation.max_steps,
        "tickets": [t.model_dump() for t in observation.tickets],
        "output_schema": schema_hint,
        "instruction": (
            "Return one JSON object with the best next action. "
            "Avoid noop unless every ticket is resolved."
        ),
    }

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        seed=seed,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are an operations agent that performs precise customer support triage.",
            },
            {
                "role": "user",
                "content": json.dumps(prompt),
            },
        ],
    )
    content = response.choices[0].message.content
    data = json.loads(content) if content else {}
    return ActionModel.model_validate(data)


def run_task(task_id: str, client: OpenAI | None, model: str, seed: int, heuristic_only: bool) -> Tuple[float, Dict[str, float], float]:
    env = SupportTriageEnv(task_id=task_id)
    observation = env.reset(task_id=task_id)
    done = False
    success = False
    step_index = 0
    reward_values: List[float] = []

    log_start(task_name=task_id, model_name=model)

    try:
        while not done:
            step_index += 1
            if heuristic_only or client is None:
                action = heuristic_action(observation)
            else:
                try:
                    action = llm_action(client=client, model=model, observation=observation, seed=seed)
                except Exception:
                    action = heuristic_action(observation)

            observation, reward, done, _info = env.step(action)
            reward_values.append(reward.value)
            log_step(step=step_index, action=action, reward=reward.value, done=done, error=None)

        success = True
    finally:
        log_end(success=success, steps=step_index, rewards=reward_values)
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    final_state = env.state()
    score, components = grade_state(final_state)
    return score, components, final_state.running_score


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Submission inference runner")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--heuristic-only", action="store_true")
    parser.add_argument("--max-runtime-seconds", type=int, default=DEFAULT_MAX_RUNTIME_SECONDS)
    args = parser.parse_args()

    api_base_url = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL)
    hf_token = os.getenv("HF_TOKEN")
    local_image_name = os.getenv("LOCAL_IMAGE_NAME")
    if hf_token is None:
        raise ValueError("HF_TOKEN environment variable is required")

    api_key = hf_token

    client = None
    if not args.heuristic_only:
        client = OpenAI(api_key=api_key, base_url=api_base_url)

    started = time.time()

    task_results: Dict[str, Dict[str, object]] = {}
    scores: List[float] = []

    for task_id in sorted(TASKS.keys()):
        elapsed = time.time() - started
        if elapsed > args.max_runtime_seconds:
            raise TimeoutError(
                f"Inference exceeded max runtime ({args.max_runtime_seconds}s) before task {task_id}"
            )

        score, components, trajectory_reward = run_task(
            task_id=task_id,
            client=client,
            model=model_name,
            seed=args.seed,
            heuristic_only=args.heuristic_only,
        )
        log_score(task_id=task_id, task_score=score, trajectory_reward=trajectory_reward)
        scores.append(score)
        task_results[task_id] = {
            "task_score": _one_decimal_score(score),
            "grade_components": components,
            "trajectory_reward": _one_decimal_score(trajectory_reward),
        }

    aggregate = sum(scores) / len(scores) if scores else 0.0
    total_runtime = round(time.time() - started, 3)
    log_summary(aggregate_score=aggregate, runtime_seconds=total_runtime)

    _ = {
        "api_base_url": api_base_url,
        "model": model_name,
        "seed": args.seed,
        "heuristic_only": args.heuristic_only,
        "runtime_seconds": total_runtime,
        "aggregate_score": _one_decimal_score(aggregate),
        "tasks": task_results,
        "local_image_name": local_image_name,
    }


if __name__ == "__main__":
    main()
