from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple

from .graders import grade_state
from .models import ActionModel, EnvironmentState, ObservationModel, RewardModel, TicketView
from .tasks import TASKS, TaskSpec


SCORE_EPS = 0.01


def _strict_score(value: float) -> float:
    return min(1.0 - SCORE_EPS, max(SCORE_EPS, value))


class SupportTriageEnv:
    """Real-world support triage simulation with OpenEnv-style API."""

    def __init__(self, task_id: str = "easy_refund_and_login"):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}")
        self._task_id = task_id
        self._task = TASKS[task_id]
        self._state: EnvironmentState | None = None
        self._action_history: List[str] = []
        self.reset(task_id=task_id)

    def _build_initial_tickets(self, task: TaskSpec) -> List[TicketView]:
        return [
            TicketView(
                ticket_id=t.ticket_id,
                subject=t.subject,
                customer_message=t.customer_message,
                customer_tier=t.customer_tier,
                order_value=t.order_value,
                priority=None,
                team=None,
                status="open",
                drafted_reply=None,
            )
            for t in task.tickets
        ]

    def reset(self, task_id: str | None = None) -> ObservationModel:
        if task_id is not None:
            if task_id not in TASKS:
                raise ValueError(f"Unknown task_id: {task_id}")
            self._task_id = task_id
            self._task = TASKS[task_id]

        self._action_history = []
        self._state = EnvironmentState(
            task_id=self._task.task_id,
            task_name=self._task.name,
            objective=self._task.objective,
            step_index=0,
            max_steps=self._task.max_steps,
            done=False,
            tickets=self._build_initial_tickets(self._task),
            event_log=["Environment reset"],
            running_score=_strict_score(0.0),
            metadata={"difficulty": self._task.difficulty},
        )
        return self._to_observation(["Start triaging tickets"])

    def state(self) -> EnvironmentState:
        if self._state is None:
            raise RuntimeError("Environment has not been initialized")
        return deepcopy(self._state)

    def _to_observation(self, recent_events: List[str]) -> ObservationModel:
        state = self.state()
        return ObservationModel(
            task_id=state.task_id,
            task_name=state.task_name,
            objective=state.objective,
            step_index=state.step_index,
            max_steps=state.max_steps,
            tickets=state.tickets,
            recent_events=recent_events,
        )

    def _find_ticket(self, ticket_id: str) -> TicketView:
        if self._state is None:
            raise RuntimeError("Environment has not been initialized")
        for ticket in self._state.tickets:
            if ticket.ticket_id == ticket_id:
                return ticket
        raise ValueError(f"Unknown ticket_id: {ticket_id}")

    def _get_ticket_spec(self, ticket_id: str):
        for spec in self._task.tickets:
            if spec.ticket_id == ticket_id:
                return spec
        raise ValueError(f"Unknown ticket_id: {ticket_id}")

    def _base_reward(self) -> Dict[str, float]:
        return {"step_penalty": -0.01}

    def _is_repeated_action(self, action: ActionModel) -> bool:
        signature = f"{action.action_type}:{action.ticket_id}:{action.priority}:{action.team}:{action.reply_text}:{action.resolution_note}"
        is_repeat = signature in self._action_history[-3:]
        self._action_history.append(signature)
        return is_repeat

    def _apply_action(self, action: ActionModel) -> Tuple[Dict[str, float], str]:
        if self._state is None:
            raise RuntimeError("Environment has not been initialized")

        components = self._base_reward()
        event = ""

        if action.action_type == "noop":
            components["noop_penalty"] = -0.03
            event = "Agent used noop"
            return components, event

        if not action.ticket_id:
            components["invalid_action"] = -0.15
            event = "Missing ticket_id"
            return components, event

        ticket = self._find_ticket(action.ticket_id)
        spec = self._get_ticket_spec(action.ticket_id)

        if action.action_type == "classify_ticket":
            if action.priority is None or action.team is None:
                components["invalid_action"] = -0.15
                event = f"Invalid classification for {ticket.ticket_id}"
                return components, event

            ticket.priority = action.priority
            ticket.team = action.team
            ticket.status = "in_progress"

            priority_match = 1.0 if action.priority == spec.expected_priority else 0.0
            team_match = 1.0 if action.team == spec.expected_team else 0.0
            components["classification"] = (priority_match * 0.12) + (team_match * 0.12) - 0.04
            event = f"Classified {ticket.ticket_id} as {action.priority}/{action.team}"
            return components, event

        if action.action_type == "draft_reply":
            text = (action.reply_text or "").strip()
            if len(text) < 20:
                components["low_quality_reply"] = -0.08
                event = f"Low-quality reply for {ticket.ticket_id}"
                return components, event

            ticket.drafted_reply = text
            keyword_hits = sum(1 for kw in spec.required_reply_keywords if kw.lower() in text.lower())
            coverage = keyword_hits / len(spec.required_reply_keywords) if spec.required_reply_keywords else 1.0
            components["reply_quality"] = (coverage * 0.20) - 0.02
            event = f"Drafted reply for {ticket.ticket_id}"
            return components, event

        if action.action_type == "resolve_ticket":
            note = (action.resolution_note or "").strip()
            if len(note) < 10:
                components["invalid_resolution"] = -0.10
                event = f"Resolution note too short for {ticket.ticket_id}"
                return components, event

            if ticket.priority is None or ticket.team is None:
                components["premature_resolution"] = -0.12
                event = f"Tried resolving {ticket.ticket_id} before classification"
                return components, event

            ticket.status = "resolved"
            components["resolution"] = 0.18
            if ticket.drafted_reply:
                components["resolution_with_reply_bonus"] = 0.07
            event = f"Resolved {ticket.ticket_id}"
            return components, event

        components["invalid_action"] = -0.20
        event = "Unsupported action type"
        return components, event

    def step(self, action: ActionModel | Dict[str, Any]):
        if self._state is None:
            raise RuntimeError("Environment has not been initialized")
        if self._state.done:
            observation = self._to_observation(["Episode already finished"])
            reward = RewardModel(value=0.0, components={}, explanation="Episode already finished")
            return observation, reward, True, {"task_score": _strict_score(self._state.running_score)}

        parsed_action = action if isinstance(action, ActionModel) else ActionModel.model_validate(action)

        components, event = self._apply_action(parsed_action)

        if self._is_repeated_action(parsed_action):
            components["loop_penalty"] = components.get("loop_penalty", 0.0) - 0.05
            event = f"{event}; repeated action detected"

        self._state.step_index += 1
        self._state.event_log.append(event)

        all_resolved = all(t.status == "resolved" for t in self._state.tickets)
        timed_out = self._state.step_index >= self._state.max_steps
        self._state.done = all_resolved or timed_out

        if self._state.done:
            task_score, grade_components = grade_state(self._state, self._task)
            completion_bonus = task_score * 0.40
            components["completion_bonus"] = completion_bonus
            self._state.metadata["grade"] = grade_components

        reward_value = round(sum(components.values()), 4)
        self._state.running_score = round(_strict_score(self._state.running_score + reward_value), 4)

        explanation = ", ".join(f"{k}={v:+.3f}" for k, v in components.items())
        reward = RewardModel(value=reward_value, components=components, explanation=explanation)

        observation = self._to_observation([event])
        info = {
            "task_id": self._state.task_id,
            "running_score": self._state.running_score,
            "task_score": round(_strict_score(self._state.metadata.get("grade", {}).get("overall", self._state.running_score)), 4),
            "done_reason": "all_resolved" if all_resolved else ("max_steps" if timed_out else None),
            "grade": self._state.metadata.get("grade", {}),
        }
        return observation, reward, self._state.done, info
