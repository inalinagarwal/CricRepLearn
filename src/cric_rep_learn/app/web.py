"""Lightweight FastAPI front door for the three product modes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cric_rep_learn.app.services import run_dream_xi, run_match_sim, run_player_dive

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="CricRepLearn", version="0.1.0")


class DreamXiRequest(BaseModel):
    team_a_name: str = "IND"
    team_b_name: str = "ENG"
    team_a_batters: str
    team_b_batters: str
    team_a_bowlers: str
    team_b_bowlers: str
    venue: str | None = "Lord's"
    date: str | None = None
    sims: int = Field(default=60, ge=10, le=400)
    max_credits: float | None = 100.0


class MatchSimRequest(BaseModel):
    first_batters: str
    first_bowlers: str
    chase_batters: str
    chase_bowlers: str
    venue: str | None = "Lord's"
    date: str | None = None
    sims: int = Field(default=80, ge=10, le=400)


class PlayerDiveRequest(BaseModel):
    batter: str
    bowlers: str
    venue: str | None = None
    max_balls: int = Field(default=120, ge=20, le=200)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "modes": ["dream_xi", "match_sim", "player_dive"]}


@app.post("/api/dream-xi")
def api_dream_xi(body: DreamXiRequest) -> dict[str, Any]:
    try:
        return run_dream_xi(**body.model_dump())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/match-sim")
def api_match_sim(body: MatchSimRequest) -> dict[str, Any]:
    try:
        return run_match_sim(**body.model_dump())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/player-dive")
def api_player_dive(body: PlayerDiveRequest) -> dict[str, Any]:
    try:
        return run_player_dive(**body.model_dump())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def serve(*, host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    print(f"CricRepLearn UI → http://{host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info")


def main() -> None:
    serve()


if __name__ == "__main__":
    main()
