from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from openenv_support_triage.environment import SupportTriageEnv
from openenv_support_triage.models import ActionModel
from openenv_support_triage.tasks import TASKS


app = FastAPI(title="OpenEnv Support Triage", version="0.1.0")
env = SupportTriageEnv()
UI_FILE = Path(__file__).resolve().parents[1] / "ui" / "index.html"


class ResetRequest(BaseModel):
    task_id: str | None = None


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    if UI_FILE.exists():
        return UI_FILE.read_text(encoding="utf-8")
    return "<h1>OpenEnv Support Triage</h1><p>UI file missing.</p>"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "tasks": sorted(list(TASKS.keys()))}


@app.post("/reset")
def reset(req: ResetRequest | None = None):
    task_id = req.task_id if req and req.task_id else env.state().task_id
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")
    return env.reset(task_id=task_id).model_dump()


@app.post("/step")
def step(action: ActionModel):
    observation, reward, done, info = env.step(action)
    score = info.get("score", info.get("task_score", info.get("running_score", 0.01)))
    return {
        "observation": observation.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
        "score": score,
        "task_score": info.get("task_score", score),
    }


@app.get("/state")
def state():
    state = env.state().model_dump()
    grade = state.get("metadata", {}).get("grade", {})
    score = grade.get("overall", 0.01)
    state["score"] = score
    state["task_score"] = score
    return state


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
