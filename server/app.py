from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from openenv_support_triage.environment import SupportTriageEnv
from openenv_support_triage.models import ActionModel
from openenv_support_triage.tasks import TASKS


app = FastAPI(title="OpenEnv Support Triage", version="0.1.0")
env = SupportTriageEnv()


class ResetRequest(BaseModel):
    task_id: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "tasks": sorted(list(TASKS.keys()))}


@app.post("/reset")
def reset(req: ResetRequest):
    if req.task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {req.task_id}")
    return env.reset(task_id=req.task_id).model_dump()


@app.post("/step")
def step(action: ActionModel):
    observation, reward, done, info = env.step(action)
    return {
        "observation": observation.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return env.state().model_dump()


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
