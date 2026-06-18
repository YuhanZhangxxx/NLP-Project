"""
HTTP wrapper around the DB scorer so it can run as a standalone Railway service.

Reuses the existing fetch/build/score/save functions from score_from_db.py —
no scoring logic is duplicated here. The Node backend will POST to /score
instead of spawning a local Python process.

Env required: DATABASE_URL, OPENAI_API_KEY
Run: uvicorn app:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make skill_detection/ importable (score_from_db, match_scorer live there)
sys.path.insert(0, str(Path(__file__).parent / "skill_detection"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from score_from_db import fetch_transcripts, build_payload, save_verdict  # noqa: E402
from match_scorer import score_match  # noqa: E402

app = FastAPI(title="FCPBRL Scorer Service")


class ScoreRequest(BaseModel):
    livestream_id: str
    save: bool = True
    force: bool = False
    skip_account_updates: bool = False
    tie_threshold: float = 0.75


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    rows = fetch_transcripts(req.livestream_id)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No transcripts for livestream {req.livestream_id}",
        )

    try:
        payload, rapper_a_id, rapper_b_id = build_payload(req.livestream_id, rows)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    payload["tie_threshold"] = req.tie_threshold

    try:
        result = score_match(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scorer failed: {e}")

    saved = False
    if req.save and result.get("status") == "ok":
        save_verdict(
            req.livestream_id,
            result,
            rapper_a_id,
            rapper_b_id,
            force=req.force,
            update_accounts=not req.skip_account_updates,
        )
        saved = True

    return {"status": result.get("status"), "saved": saved, "result": result}
