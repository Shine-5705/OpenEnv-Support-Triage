from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openenv_support_triage.environment import SupportTriageEnv
from openenv_support_triage.graders import grade_state
from openenv_support_triage.models import ActionModel, ObservationModel
from openenv_support_triage.tasks import TASKS


DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_SEED = 7
SCORE_EPS = 0.0001


def strict_score(value: float) -> float:
    return min(1.0 - SCORE_EPS, max(SCORE_EPS, value))


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
            "Return only one JSON object. Choose a single best next action. "
            "Avoid noop unless everything is resolved."
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
                "content": "You are an operations agent that performs customer support triage precisely.",
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


def run_task(task_id: str, model: str, seed: int, use_heuristic_only: bool = False) -> Tuple[float, Dict[str, float], float]:
    env = SupportTriageEnv(task_id=task_id)
    observation = env.reset(task_id=task_id)

    client = None if use_heuristic_only else OpenAI()

    done = False
    while not done:
        if use_heuristic_only:
            action = heuristic_action(observation)
        else:
            try:
                action = llm_action(client=client, model=model, observation=observation, seed=seed)
            except Exception:
                action = heuristic_action(observation)

        observation, reward, done, _ = env.step(action)

    final_state = env.state()
    task_score, components = grade_state(final_state)
    return task_score, components, final_state.running_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible OpenEnv baseline inference")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--heuristic-only", action="store_true")
    args = parser.parse_args()

    if not args.heuristic_only and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required unless --heuristic-only is set")

    results = {}
    scores = []

    for task_id in sorted(TASKS.keys()):
        score, components, running_score = run_task(
            task_id=task_id,
            model=args.model,
            seed=args.seed,
            use_heuristic_only=args.heuristic_only,
        )
        scores.append(score)
        results[task_id] = {
            "task_score": round(strict_score(score), 4),
            "grade_components": components,
            "trajectory_reward": round(strict_score(running_score), 4),
        }

    aggregate = sum(scores) / len(scores) if scores else 0.0
    payload = {
        "model": args.model,
        "seed": args.seed,
        "heuristic_only": args.heuristic_only,
        "aggregate_score": round(strict_score(aggregate), 4),
        "tasks": results,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
