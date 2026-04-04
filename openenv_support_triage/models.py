from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TicketView(BaseModel):
    ticket_id: str
    subject: str
    customer_message: str
    customer_tier: Literal["standard", "premium", "enterprise"]
    order_value: float = 0.0
    priority: Optional[Literal["low", "medium", "high", "urgent"]] = None
    team: Optional[Literal["support", "billing", "technical", "risk"]] = None
    status: Literal["open", "in_progress", "resolved"] = "open"
    drafted_reply: Optional[str] = None


class ObservationModel(BaseModel):
    task_id: str
    task_name: str
    objective: str
    step_index: int
    max_steps: int
    tickets: List[TicketView]
    recent_events: List[str] = Field(default_factory=list)


class ActionModel(BaseModel):
    action_type: Literal["classify_ticket", "draft_reply", "resolve_ticket", "noop"]
    ticket_id: Optional[str] = None
    priority: Optional[Literal["low", "medium", "high", "urgent"]] = None
    team: Optional[Literal["support", "billing", "technical", "risk"]] = None
    reply_text: Optional[str] = None
    resolution_note: Optional[str] = None


class RewardModel(BaseModel):
    value: float
    components: Dict[str, float] = Field(default_factory=dict)
    explanation: str


class EnvironmentState(BaseModel):
    task_id: str
    task_name: str
    objective: str
    step_index: int
    max_steps: int
    done: bool
    tickets: List[TicketView]
    event_log: List[str]
    running_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
