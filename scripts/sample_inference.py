from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openenv_support_triage.environment import SupportTriageEnv
from openenv_support_triage.models import ActionModel, ObservationModel

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
TEMPERATURE = 0
MAX_TOKENS = 300
MAX_STEPS = 20
TASK_ID = os.getenv("TASK_ID", "easy_refund_and_login")

SYSTEM_PROMPT = (
    "You are a customer support operations agent. "
    "Pick exactly one next action in JSON to maximize triage quality and completion speed."
)

FALLBACK_ACTION: Dict[str, Any] = {
    "action_type": "noop",
    "ticket_id": None,
    "priority": None,
    "team": None,
    "reply_text": None,
    "resolution_note": None,
}


def build_user_prompt(step: int, observation: ObservationModel, history: List[str]) -> str:
    payload = {
        "step": step,
        "objective": observation.objective,
        "step_index": observation.step_index,
        "max_steps": observation.max_steps,
        "tickets": [ticket.model_dump() for ticket in observation.tickets],
        "history": history[-5:],
        "required_output_json": {
            "action_type": "classify_ticket|draft_reply|resolve_ticket|noop",
            "ticket_id": "ticket id or null",
            "priority": "low|medium|high|urgent or null",
            "team": "support|billing|technical|risk or null",
            "reply_text": "string or null",
            "resolution_note": "string or null",
        },
    }
    return json.dumps(payload)


def parse_model_action(response_text: str) -> Dict[str, Any]:
    try:
        data = json.loads(response_text)
        action = ActionModel.model_validate(data)
        return action.model_dump()
    except Exception:
        return FALLBACK_ACTION


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required")

    client = OpenAI()
    env = SupportTriageEnv(task_id=TASK_ID)
    history: List[str] = []

    try:
        observation = env.reset(task_id=TASK_ID)
        done = False
        print(f"Episode goal: {observation.objective}")

        for step in range(1, MAX_STEPS + 1):
            if done:
                print("Environment signalled done. Stopping early.")
                break

            user_prompt = build_user_prompt(step, observation, history)
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ]

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    response_format={"type": "json_object"},
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:
                print(f"Model request failed ({exc}). Using fallback action.")
                response_text = json.dumps(FALLBACK_ACTION)

            action_payload = parse_model_action(response_text)
            print(f"Step {step}: model suggested -> {action_payload}")

            observation, reward, done, info = env.step(ActionModel.model_validate(action_payload))
            history_line = (
                f"Step {step}: {action_payload} -> reward {reward.value:+.2f}, done={done}, "
                f"running_score={info.get('running_score')}"
            )
            history.append(history_line)
            print(f"  Reward: {reward.value:+.2f} | Done: {done} | Info: {info}")

            if done:
                print("Episode complete.")
                break
        else:
            print(f"Reached max steps ({MAX_STEPS}).")

        final_state = env.state()
        print("Final state summary:")
        print(json.dumps(final_state.model_dump(), indent=2))

    finally:
        # No explicit close() on this environment; this mirrors resource cleanup intent.
        pass


if __name__ == "__main__":
    main()
