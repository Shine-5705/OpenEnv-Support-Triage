#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_BUILD_TIMEOUT="${DOCKER_BUILD_TIMEOUT:-1200}"
INFERENCE_TIMEOUT="${INFERENCE_TIMEOUT:-1200}"
SPACE_URL="${SPACE_URL:-}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

log() {
  printf "%b\n" "$*"
}

pass() {
  printf "%b\n" "${GREEN}PASS${NC}: $*"
}

fail() {
  printf "%b\n" "${RED}FAIL${NC}: $*"
}

hint() {
  printf "%b\n" "${YELLOW}HINT${NC}: $*"
}

stop_at() {
  printf "\n"
  printf "%b\n" "${BOLD}Stopped at ${1}.${NC}"
  exit 1
}

run_with_timeout() {
  local timeout_seconds="$1"
  shift

  if command -v timeout >/dev/null 2>&1; then
    timeout "$timeout_seconds" "$@"
  else
    "$@"
  fi
}

log "${BOLD}Step 1/6: Python + inference script sanity${NC} ..."

if ! command -v python >/dev/null 2>&1; then
  fail "python command not found"
  hint "Install Python 3.11+ and activate your virtual environment"
  stop_at "Step 1"
fi

if (cd "$REPO_DIR" && python -c "import openenv_support_triage, app" >/dev/null 2>&1); then
  pass "Python imports succeeded"
else
  fail "Python import sanity failed"
  hint "Run: pip install -r requirements.txt"
  stop_at "Step 1"
fi

if [ -f "$REPO_DIR/inference.py" ]; then
  pass "Root inference.py found"
else
  fail "inference.py not found in repo root"
  stop_at "Step 1"
fi

if (cd "$REPO_DIR" && python -m compileall inference.py >/dev/null 2>&1); then
  pass "inference.py compiles"
else
  fail "inference.py has syntax issues"
  stop_at "Step 1"
fi

log "${BOLD}Step 2/6: Running docker build${NC} ..."

if ! command -v docker >/dev/null 2>&1; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/ directory"
  stop_at "Step 2"
fi

log "  Found Dockerfile in $DOCKER_CONTEXT"

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build -t openenv-support-triage "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

log "${BOLD}Step 3/6: Running openenv validate${NC} ..."

if ! command -v openenv >/dev/null 2>&1; then
  fail "openenv command not found"
  hint "Install it: pip install openenv-core"
  stop_at "Step 3"
fi

log "${BOLD}Step 4/6: Endpoint smoke test (step/reset/state)${NC} ..."

if ! command -v curl >/dev/null 2>&1; then
  fail "curl command not found"
  hint "Install curl and retry"
  stop_at "Step 4"
fi

HEALTH_JSON=$(curl -sS "http://127.0.0.1:7860/health" || true)
if [ -n "$HEALTH_JSON" ]; then
  pass "Local health endpoint reachable on :7860"
else
  fail "Local health endpoint is not reachable"
  hint "Start server: python -m uvicorn app:app --host 0.0.0.0 --port 7860"
  stop_at "Step 4"
fi

RESET_JSON=$(curl -sS -X POST "http://127.0.0.1:7860/reset" -H "Content-Type: application/json" -d '{"task_id":"easy_refund_and_login"}' || true)
if printf "%s" "$RESET_JSON" | grep -q '"task_id"'; then
  pass "reset() responded"
else
  fail "reset() did not return expected payload"
  stop_at "Step 4"
fi

STEP_JSON=$(curl -sS -X POST "http://127.0.0.1:7860/step" -H "Content-Type: application/json" -d '{"action_type":"classify_ticket","ticket_id":"T-1001","priority":"medium","team":"billing"}' || true)
if printf "%s" "$STEP_JSON" | grep -q '"reward"'; then
  pass "step() responded"
else
  fail "step() did not return expected payload"
  stop_at "Step 4"
fi

STATE_JSON=$(curl -sS "http://127.0.0.1:7860/state" || true)
if printf "%s" "$STATE_JSON" | grep -q '"step_index"'; then
  pass "state() responded"
else
  fail "state() did not return expected payload"
  stop_at "Step 4"
fi

log "${BOLD}Step 5/6: Baseline/inference reproducibility${NC} ..."

INFERENCE_OK=false
INFERENCE_OUTPUT=$(cd "$REPO_DIR" && run_with_timeout "$INFERENCE_TIMEOUT" python inference.py --heuristic-only --seed 7 2>&1) && INFERENCE_OK=true

if [ "$INFERENCE_OK" = true ]; then
  pass "inference.py completed"
  if printf "%s" "$INFERENCE_OUTPUT" | grep -q '"aggregate_score"'; then
    pass "inference.py produced scores"
  else
    fail "inference output missing aggregate_score"
    printf "%s\n" "$INFERENCE_OUTPUT"
    stop_at "Step 5"
  fi
else
  fail "inference.py failed or timed out (timeout=${INFERENCE_TIMEOUT}s)"
  printf "%s\n" "$INFERENCE_OUTPUT"
  stop_at "Step 5"
fi

log "${BOLD}Step 6/6: Task graders and HF Space ping${NC} ..."

GRADER_CHECK=$(cd "$REPO_DIR" && python - <<'PY'
from openenv_support_triage.environment import SupportTriageEnv
from openenv_support_triage.graders import grade_state
from openenv_support_triage.tasks import TASKS

if len(TASKS) < 3:
    raise SystemExit("Need at least 3 tasks")

for task_id in sorted(TASKS.keys()):
    env = SupportTriageEnv(task_id=task_id)
    env.reset(task_id=task_id)
    score, _components = grade_state(env.state())
    if not (0.0 <= score <= 1.0):
        raise SystemExit(f"Score out of range for {task_id}: {score}")

print("OK")
PY
)

if [ "$GRADER_CHECK" = "OK" ]; then
  pass "3+ tasks exist and grader scores are within 0.0-1.0"
else
  fail "grader range check failed"
  printf "%s\n" "$GRADER_CHECK"
  stop_at "Step 6"
fi

if [ -n "$SPACE_URL" ]; then
  HTTP_CODE=$(curl -o /tmp/space_ping.out -sS -w "%{http_code}" "$SPACE_URL" || true)
  if [ "$HTTP_CODE" = "200" ]; then
    pass "HF Space URL returned 200"
  else
    fail "HF Space URL did not return 200 (got $HTTP_CODE)"
    stop_at "Step 6"
  fi

  SPACE_RESET_CODE=$(curl -o /tmp/space_reset.out -sS -w "%{http_code}" -X POST "${SPACE_URL%/}/reset" -H "Content-Type: application/json" -d '{"task_id":"easy_refund_and_login"}' || true)
  if [ "$SPACE_RESET_CODE" = "200" ]; then
    pass "HF Space reset() returned 200"
  else
    fail "HF Space reset() did not return 200 (got $SPACE_RESET_CODE)"
    stop_at "Step 6"
  fi
else
  hint "SPACE_URL not set; skipping HF ping/reset checks"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate . 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

printf "\n"
printf "%b\n" "${BOLD}========================================${NC}"
printf "%b\n" "${GREEN}${BOLD}  All checks passed!${NC}"
printf "%b\n" "${GREEN}${BOLD}  Your submission is ready to submit.${NC}"
printf "%b\n" "${BOLD}========================================${NC}"
printf "\n"

exit 0
