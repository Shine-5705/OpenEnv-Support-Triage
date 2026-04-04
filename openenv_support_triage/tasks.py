from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field


class TicketSpec(BaseModel):
    ticket_id: str
    subject: str
    customer_message: str
    customer_tier: Literal["standard", "premium", "enterprise"]
    order_value: float = 0.0
    expected_priority: Literal["low", "medium", "high", "urgent"]
    expected_team: Literal["support", "billing", "technical", "risk"]
    required_reply_keywords: List[str] = Field(default_factory=list)


class TaskSpec(BaseModel):
    task_id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
    tickets: List[TicketSpec]
    max_steps: int = 40


TASKS: Dict[str, TaskSpec] = {
    "easy_refund_and_login": TaskSpec(
        task_id="easy_refund_and_login",
        name="Refund + Login Assistance",
        difficulty="easy",
        objective=(
            "Classify and resolve two inbound tickets: one billing refund request and "
            "one technical login issue."
        ),
        max_steps=18,
        tickets=[
            TicketSpec(
                ticket_id="T-1001",
                subject="Need refund for duplicate charge",
                customer_message=(
                    "I was charged twice for order #8841. Please refund one of the charges."
                ),
                customer_tier="standard",
                order_value=79.99,
                expected_priority="medium",
                expected_team="billing",
                required_reply_keywords=["refund", "duplicate", "48 hours"],
            ),
            TicketSpec(
                ticket_id="T-1002",
                subject="Cannot log in after password reset",
                customer_message=(
                    "I reset my password but still cannot access my account. I need help today."
                ),
                customer_tier="premium",
                order_value=0.0,
                expected_priority="high",
                expected_team="technical",
                required_reply_keywords=["verify", "reset link", "support"],
            ),
        ],
    ),
    "medium_fraud_shipping_invoice": TaskSpec(
        task_id="medium_fraud_shipping_invoice",
        name="Fraud, Shipping, and Invoice Mix",
        difficulty="medium",
        objective=(
            "Handle three mixed-priority tickets with correct triage and customer-safe responses."
        ),
        max_steps=28,
        tickets=[
            TicketSpec(
                ticket_id="T-2001",
                subject="Unknown purchase on my card",
                customer_message=(
                    "I see a purchase I did not make. Please stop any further charges now."
                ),
                customer_tier="premium",
                order_value=420.00,
                expected_priority="urgent",
                expected_team="risk",
                required_reply_keywords=["freeze", "verify", "fraud"],
            ),
            TicketSpec(
                ticket_id="T-2002",
                subject="Package delayed by 9 days",
                customer_message=(
                    "Tracking has not updated for over a week. Is this lost?"
                ),
                customer_tier="standard",
                order_value=54.20,
                expected_priority="medium",
                expected_team="support",
                required_reply_keywords=["carrier", "update", "replacement"],
            ),
            TicketSpec(
                ticket_id="T-2003",
                subject="Invoice has wrong company name",
                customer_message=(
                    "Please correct the legal entity name and resend invoice INV-332."
                ),
                customer_tier="enterprise",
                order_value=1300.00,
                expected_priority="high",
                expected_team="billing",
                required_reply_keywords=["invoice", "legal", "resend"],
            ),
        ],
    ),
    "hard_enterprise_outage_bundle": TaskSpec(
        task_id="hard_enterprise_outage_bundle",
        name="Enterprise Outage + Policy Edge Cases",
        difficulty="hard",
        objective=(
            "Process four tickets where urgency, customer tier, and policy constraints all matter."
        ),
        max_steps=40,
        tickets=[
            TicketSpec(
                ticket_id="T-3001",
                subject="Production API returns 500 for all requests",
                customer_message=(
                    "Our checkout is down in production. This affects all customers globally."
                ),
                customer_tier="enterprise",
                order_value=0.0,
                expected_priority="urgent",
                expected_team="technical",
                required_reply_keywords=["incident", "status page", "escalated"],
            ),
            TicketSpec(
                ticket_id="T-3002",
                subject="Chargeback warning from bank",
                customer_message=(
                    "A bank warned us about potential chargeback abuse from recent transactions."
                ),
                customer_tier="enterprise",
                order_value=9800.00,
                expected_priority="urgent",
                expected_team="risk",
                required_reply_keywords=["risk", "review", "freeze"],
            ),
            TicketSpec(
                ticket_id="T-3003",
                subject="Need prorated downgrade this month",
                customer_message=(
                    "We want to downgrade seats mid-cycle and receive prorated billing adjustment."
                ),
                customer_tier="premium",
                order_value=2300.00,
                expected_priority="high",
                expected_team="billing",
                required_reply_keywords=["prorated", "billing", "effective"],
            ),
            TicketSpec(
                ticket_id="T-3004",
                subject="Content moderation false positive report",
                customer_message=(
                    "Our harmless listing was automatically removed. Please review and restore."
                ),
                customer_tier="standard",
                order_value=110.00,
                expected_priority="medium",
                expected_team="support",
                required_reply_keywords=["review", "restore", "policy"],
            ),
        ],
    ),
}


def list_tasks() -> List[TaskSpec]:
    return [TASKS[key] for key in sorted(TASKS.keys())]
