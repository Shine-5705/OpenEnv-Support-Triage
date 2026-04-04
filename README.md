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

A real-world OpenEnv environment for evaluating agent behavior in customer support operations. The environment simulates triage work across billing, technical support, risk, and general support queues.

## Why This Environment Matters

Customer support triage is a production workflow teams run every day. High-quality agents must:

- infer urgency from free-form customer text
- route tickets to the right team
- draft actionable, policy-aware replies
- resolve tickets efficiently without taking premature or destructive actions

This environment is designed for both training and evaluation with dense rewards and deterministic grading.

## OpenEnv API Compliance

Implemented interfaces and models:

- typed Pydantic models for observation, action, and reward
- `reset(task_id)` -> initial observation
- `step(action)` -> `(observation, reward, done, info)`
- `state()` -> full internal state snapshot
- `openenv.yaml` metadata for discovery and validation

Entrypoint: `openenv_support_triage.environment:SupportTriageEnv`

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

`ObservationModel` contains:

- task metadata (`task_id`, `task_name`, `objective`)
- trajectory progress (`step_index`, `max_steps`)
- current queue (`tickets[]` with status, priority/team assignments, drafted replies)
- `recent_events` for immediate feedback

## Reward Function

Reward is shaped at each step (not just terminal):

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

Each task has a deterministic grader returning a score in [0.0, 1.0] using:

- classification quality
- reply quality
- resolution completeness

## Project Layout

- `inference.py`: required root inference script for submission
- `openenv.yaml`: OpenEnv metadata
- `openenv_support_triage/`: environment implementation and typed models
- `server/app.py`: deployment app entrypoint
- `server/cli.py`: script entrypoint for server mode
- `scripts/pre_validate.sh`: pre-submission validator

## Environment Variables

Required for model-backed inference:

- `API_BASE_URL` (default: `https://api.openai.com/v1`)
- `MODEL_NAME` (default: `gpt-4.1-mini`)
- `HF_TOKEN` (no default)

Optional:

- `LOCAL_IMAGE_NAME` (only needed if your workflow uses local docker image loading)

All LLM calls use OpenAI client:

`from openai import OpenAI`

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Local Usage

Run API service (local):

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Smoke test:

```bash
curl http://localhost:7860/health
```

## Inference Script (Submission Contract)

The required submission script is `inference.py` in repository root.

Structured stdout logs are emitted as:

- `START {...}`
- `STEP {...}`
- `END {...}`

This satisfies required logging format.

Run heuristic deterministic baseline:

```bash
python inference.py --heuristic-only --seed 7
```

Run model-backed inference:

```bash
python inference.py --seed 7
```

## Baseline Inference (Legacy Helper)

Set credentials:

```bash
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4.1-mini
set HF_TOKEN=your_token_here
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

## Baseline Scores

Expected deterministic reference with `--heuristic-only`:

- `easy_refund_and_login`: 0.8000
- `medium_fraud_shipping_invoice`: 0.7667
- `hard_enterprise_outage_bundle`: 0.7250
- aggregate: 0.7639

OpenAI-driven scores vary by model family but are reproducible run-to-run for a fixed model and seed when deterministic decoding is used.

## Docker

Build and run container:

```bash
docker build -t openenv-support-triage .
docker run --rm -p 7860:7860 openenv-support-triage
```

## Hugging Face Spaces Deployment

1. Create a new Space and choose `Docker` SDK.
2. Push this repository to the Space.
3. Confirm `README.md` front matter includes `tags: [openenv]`.
4. Space will auto-build from `Dockerfile` and serve on port `7860`.

## OpenEnv Validation

Run validator:

```bash
openenv validate .
```

This validates metadata and API compatibility.

## Pre-Submission Validator

Run pre-submit checks:

```bash
bash scripts/pre_validate.sh
```

Optional for HF checks:

```bash
export SPACE_URL=https://your-space-name.hf.space
bash scripts/pre_validate.sh
```

## Final Checklist

- `openenv validate .` returns OK
- `inference.py` runs and prints START/STEP/END logs
- docker build succeeds
- local `/health` and `/reset` return HTTP 200
- HF Space deploys and responds successfully
