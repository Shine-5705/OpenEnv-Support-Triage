---
title: OpenEnv Support Triage
emoji: "🤖"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - customer-support
  - agent-evals
  - llm
  - openai
  - fastapi
  - docker
---

# OpenEnv Support Triage

A complete, real-world OpenEnv environment that simulates customer support ticket triage for billing, technical, risk, and general support operations.

## Motivation

Customer support triage is a real operational workflow performed by humans every day. Agents must:

- infer urgency from free-form customer text
- route tickets to the right team
- draft actionable, policy-aware replies
- resolve tickets efficiently without taking premature or destructive actions

This environment is designed for agent training and evaluation with meaningful intermediate reward signals and deterministic final grading.

## OpenEnv API Compliance

The environment implements:

- typed Pydantic models for observation, action, and reward
- `reset(task_id)` -> initial observation
- `step(action)` -> `(observation, reward, done, info)`
- `state()` -> full internal state snapshot
- `openenv.yaml` metadata for discovery and validation

Implementation entrypoint: `openenv_support_triage.environment:SupportTriageEnv`

## Action Space

`ActionModel` supports:

- `classify_ticket`: set `priority` and `team` for a ticket
- `draft_reply`: add customer-facing response text
- `resolve_ticket`: close ticket with a resolution note
- `noop`: no operation (penalized)

Fields:

- `ticket_id`: target ticket for non-noop actions
- `priority`: one of `low|medium|high|urgent`
- `team`: one of `support|billing|technical|risk`
- `reply_text`: outbound response body
- `resolution_note`: internal closure note

## Observation Space

`ObservationModel` includes:

- task metadata (`task_id`, `task_name`, `objective`)
- trajectory progress (`step_index`, `max_steps`)
- current queue (`tickets[]` with status, priority/team assignments, drafted replies)
- `recent_events` for immediate feedback

## Reward Function

Reward is shaped per step (not only terminal):

- step cost: small negative to encourage efficiency
- positive partial reward for correct classification decisions
- reply quality reward based on required keyword coverage
- resolution reward, with bonus when reply exists
- penalties for invalid actions, noop abuse, premature resolution, and repeated loops
- terminal completion bonus weighted by deterministic grader score

This gives dense progress signals and discourages exploitative behavior.

## Tasks and Difficulty

1. `easy_refund_and_login` (easy): 2 tickets, straightforward billing + technical triage
2. `medium_fraud_shipping_invoice` (medium): 3 mixed tickets including urgent fraud handling
3. `hard_enterprise_outage_bundle` (hard): 4 high-stakes tickets with outage, risk, billing, and policy edge cases

Each task has a deterministic grader scoring 0.0-1.0 by combining:

- classification quality
- reply quality
- resolution completeness

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Local Usage

Run API service:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Quick API check:

```bash
curl http://localhost:7860/health
```

## Demo UI

A professional browser UI is available at the root route:

```bash
http://localhost:7860/
```

UI capabilities:

- task selection and reset
- manual step execution (`classify_ticket`, `draft_reply`, `resolve_ticket`, `noop`)
- live reward and running score display
- ticket board with current state
- event log for trajectory inspection

## Inference Contract (Submission)

Root script:

```bash
python inference.py --seed 7
```

Required environment variables used by `inference.py`:

- `API_BASE_URL` (default set)
- `MODEL_NAME` (default set)
- `HF_TOKEN` (no default)

Optional variable:

- `LOCAL_IMAGE_NAME` (used only for docker-image based local runners)

All LLM calls are made with OpenAI client:

```python
from openai import OpenAI
```

Stdout uses structured records with exact markers:

- `START {...}`
- `STEP {...}`
- `END {...}`

## Baseline Inference (OpenAI API)

Set credentials:

```bash
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4.1-mini
set HF_TOKEN=your_token_here
```

Run reproducible baseline over all tasks using the required root script:

```bash
python inference.py --seed 7
```

Legacy baseline runner:

```bash
python scripts/baseline_inference.py --seed 7
```

Heuristic-only fallback:

```bash
python scripts/baseline_inference.py --heuristic-only
```

The script prints per-task and aggregate scores in JSON.

The script prints per-task and aggregate scores in JSON.

## Baseline Scores

Expected deterministic reference with `--heuristic-only`:

- `easy_refund_and_login`: 0.8000
- `medium_fraud_shipping_invoice`: 0.7667
- `hard_enterprise_outage_bundle`: 0.7250
- aggregate: 0.7639

OpenAI-driven scores vary by model family but are reproducible run-to-run for a fixed model and seed when deterministic decoding is used.

## Docker

Build and run:

```bash
docker build -t openenv-support-triage .
docker run --rm -p 7860:7860 openenv-support-triage
```

## Hugging Face Spaces Deployment

1. Create a new Space and choose `Docker` SDK.
2. Push this repository to the Space.
3. Set Space variables/secrets: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`.
4. Space will auto-build from `Dockerfile` and serve on port `7860`.
5. Verify `GET /health` and `POST /reset` return HTTP 200.

## OpenEnv Validation

If you have the OpenEnv CLI installed:

```bash
openenv validate .
```

This validates metadata and API compatibility.

## Pre-Submission Validator

Run the repository validator before submitting:

```bash
bash scripts/pre_validate.sh
```

Optional for HF checks:

```bash
export SPACE_URL=https://your-space-name.hf.space
bash scripts/pre_validate.sh
```

## Final Checklist

- `openenv validate .` returns ready for multi-mode deployment
- `python inference.py --heuristic-only --seed 7` completes and prints `END`
- `docker build -t openenv-support-triage .` succeeds
- local API responds on `/health`, `/reset`, `/step`, and `/state`
- HF Space responds with 200 on `/health` and `/reset`
