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


def structured_log(tag: str, payload: Dict[str, object]) -> None:
    print(f"{tag} {json.dumps(payload, sort_keys=True)}")


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
    step_index = 0
    while not done:
        step_index += 1
        if heuristic_only or client is None:
            action = heuristic_action(observation)
        else:
            try:
                action = llm_action(client=client, model=model, observation=observation, seed=seed)
            except Exception:
                action = heuristic_action(observation)

        observation, reward, done, info = env.step(action)
        structured_log(
            "STEP",
            {
                "task_id": task_id,
                "step": step_index,
                "action_type": action.action_type,
                "ticket_id": action.ticket_id,
                "reward": round(reward.value, 4),
                "done": done,
                "running_score": info.get("running_score"),
            },
        )

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
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    api_key = hf_token or openai_api_key

    structured_log(
        "START",
        {
            "api_base_url": api_base_url,
            "model": model_name,
            "seed": args.seed,
            "heuristic_only": args.heuristic_only,
            "local_image_name": local_image_name,
        },
    )

    if not args.heuristic_only and not api_key:
        raise EnvironmentError("Set HF_TOKEN (or OPENAI_API_KEY) for model inference")

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
        scores.append(score)
        task_results[task_id] = {
            "task_score": round(score, 4),
            "grade_components": components,
            "trajectory_reward": round(trajectory_reward, 4),
        }

    aggregate = sum(scores) / len(scores) if scores else 0.0
    total_runtime = round(time.time() - started, 3)

    payload = {
        "api_base_url": api_base_url,
        "model": model_name,
        "seed": args.seed,
        "heuristic_only": args.heuristic_only,
        "runtime_seconds": total_runtime,
        "aggregate_score": round(aggregate, 4),
        "tasks": task_results,
    }
    structured_log("END", payload)


if __name__ == "__main__":
    main()
