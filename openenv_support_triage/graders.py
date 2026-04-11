from __future__ import annotations

from typing import Dict, Tuple

from .models import EnvironmentState
from .tasks import TASKS, TaskSpec


PRIORITY_POINTS = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "urgent": 4,
}

SCORE_EPS = 0.1


def _priority_match_score(expected: str, actual: str | None) -> float:
    if actual is None:
        return 0.0
    if actual == expected:
        return 1.0
    delta = abs(PRIORITY_POINTS[actual] - PRIORITY_POINTS[expected])
    if delta == 1:
        return 0.5
    return 0.0


def _reply_keyword_coverage(reply_text: str | None, keywords: list[str]) -> float:
    if not reply_text:
        return 0.0
    if not keywords:
        return 1.0
    text = reply_text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text)
    return hits / len(keywords)


def _strict_open_interval(value: float) -> float:
    return min(1.0 - SCORE_EPS, max(SCORE_EPS, value))


def _one_decimal_score(value: float) -> float:
    return round(_strict_open_interval(value), 1)


def grade_state(state: EnvironmentState, task: TaskSpec | None = None) -> Tuple[float, Dict[str, float]]:
    task_spec = task if task is not None else TASKS[state.task_id]

    ticket_count = len(task_spec.tickets)
    if ticket_count == 0:
        floor = round(SCORE_EPS, 1)
        return floor, {"classification": floor, "reply_quality": floor, "resolution": floor, "overall": floor}

    observed = {t.ticket_id: t for t in state.tickets}

    classification_total = 0.0
    reply_total = 0.0
    resolution_total = 0.0

    for spec in task_spec.tickets:
        ticket = observed[spec.ticket_id]

        priority_score = _priority_match_score(spec.expected_priority, ticket.priority)
        team_score = 1.0 if ticket.team == spec.expected_team else 0.0
        classification_total += (priority_score * 0.6) + (team_score * 0.4)

        reply_total += _reply_keyword_coverage(ticket.drafted_reply, spec.required_reply_keywords)

        is_resolved = 1.0 if ticket.status == "resolved" else 0.0
        resolution_total += is_resolved

    classification = _one_decimal_score(classification_total / ticket_count)
    reply_quality = _one_decimal_score(reply_total / ticket_count)
    resolution = _one_decimal_score(resolution_total / ticket_count)

    overall = (classification * 0.4) + (reply_quality * 0.3) + (resolution * 0.3)
    overall = _one_decimal_score(overall)
    components = {
        "classification": round(classification, 1),
        "reply_quality": round(reply_quality, 1),
        "resolution": round(resolution, 1),
        "overall": round(overall, 1),
    }
    return components["overall"], components
