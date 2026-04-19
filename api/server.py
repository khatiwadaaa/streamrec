"""
api/server.py
FastAPI serving layer for StreamRec.
Endpoints:
  GET  /health                    → uptime check (SRE pattern)
  GET  /recommend                 → top-N recommendations for a user
  POST /feedback                  → log implicit user feedback (A/B logging)
  GET  /stats/top-movies          → most-rated movies (SQL aggregation demo)
  GET  /stats/user/{user_id}      → per-user rating summary
"""

import sqlite3
import time
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from models.cf import recommend_cf
from models.mf import load_model, recommend_mf
from models.ranker import rank

DB_PATH   = "data/streamrec.db"
GO_CACHE  = "http://localhost:8080"   # Go microservice (optional)
START_TIME = time.time()

# ── shared state loaded once at startup ──────────────────────────────────────

mf_model: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load MF model cache into memory at startup
    global mf_model
    try:
        mf_model = load_model()
        print("MF model loaded into memory.")
    except FileNotFoundError:
        print("WARNING: No MF model cache found. Run `python models/mf.py` first.")
    yield


app = FastAPI(
    title="StreamRec",
    description="Content recommendation engine — TikTok infra demo",
    version="1.0.0",
    lifespan=lifespan,
)


# ── DB helper ─────────────────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["SRE"])
def health():
    """SRE uptime endpoint. Returns server status and uptime in seconds."""
    return {
        "status":   "ok",
        "uptime_s": round(time.time() - START_TIME, 1),
        "model_loaded": bool(mf_model),
    }


# ── /recommend ────────────────────────────────────────────────────────────────

@app.get("/recommend", tags=["Recommendations"])
async def recommend(
    user_id: int = Query(..., description="User to generate recommendations for"),
    n:       int = Query(10,  ge=1, le=50, description="Number of results"),
    model:   Literal["hybrid", "cf", "mf"] = Query(
        "hybrid", description="Which model to use"
    ),
):
    """
    Returns top-N movie recommendations for a user.

    - **hybrid**: merges CF + MF with weighted re-ranking (default)
    - **cf**: user-based collaborative filtering only
    - **mf**: matrix factorization only

    Results are enriched with movie titles from the DB.
    Checks Go cache first (if running) before computing.
    """
    # 1. Try Go cache first
    cache_key = f"{user_id}:{n}:{model}"
    try:
        async with httpx.AsyncClient(timeout=0.2) as client:
            resp = await client.get(f"{GO_CACHE}/cache/{cache_key}")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass   # cache miss or Go service not running — compute normally

    # 2. Generate recommendations
    if model == "cf":
        candidates = recommend_cf(user_id, n=n * 2)
    elif model == "mf":
        candidates = recommend_mf(user_id, n=n * 2, model=mf_model or None)
    else:  # hybrid
        cf_cands = recommend_cf(user_id, n=n * 2)
        mf_cands = recommend_mf(user_id, n=n * 2, model=mf_model or None)
        candidates = rank(cf_cands, mf_cands, n=n)

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=f"No recommendations found for user_id={user_id}. "
                   "Check that the user exists in the dataset."
        )

    # 3. Enrich with movie titles
    movie_ids = [c["movie_id"] for c in candidates]
    conn      = get_db()
    placeholders = ",".join("?" * len(movie_ids))
    rows = conn.execute(
        f"SELECT movie_id, title, genres FROM movies WHERE movie_id IN ({placeholders})",
        movie_ids,
    ).fetchall()
    conn.close()

    title_map = {r["movie_id"]: {"title": r["title"], "genres": r["genres"]} for r in rows}
    results = []
    for c in candidates[:n]:
        meta = title_map.get(c["movie_id"], {})
        results.append({**c, **meta})

    payload = {"user_id": user_id, "model": model, "count": len(results), "results": results}

    # 4. Write result to Go cache (fire-and-forget)
    try:
        async with httpx.AsyncClient(timeout=0.2) as client:
            await client.post(f"{GO_CACHE}/cache/{cache_key}", json=payload)
    except Exception:
        pass

    return payload


# ── /feedback ─────────────────────────────────────────────────────────────────

class FeedbackPayload(BaseModel):
    user_id:  int
    movie_id: int
    event:    Literal["click", "skip", "watch"]
    model:    Literal["cf", "mf", "hybrid"] = "hybrid"


@app.post("/feedback", status_code=201, tags=["A/B Logging"])
def feedback(body: FeedbackPayload):
    """
    Logs an implicit feedback event from the frontend.
    Persisted to the ab_log table for offline analysis.

    Events:
    - **click**: user tapped on the recommendation
    - **watch**: user watched > 70% of the content
    - **skip**: user swiped past
    """
    conn = get_db()
    conn.execute(
        "INSERT INTO ab_log (user_id, movie_id, event, model) VALUES (?,?,?,?)",
        (body.user_id, body.movie_id, body.event, body.model),
    )
    conn.commit()
    conn.close()
    return {"status": "logged", **body.model_dump()}


# ── /stats/top-movies ─────────────────────────────────────────────────────────

@app.get("/stats/top-movies", tags=["Analytics"])
def top_movies(limit: int = Query(10, ge=1, le=100)):
    """
    Returns the most-rated movies in the dataset.
    Demonstrates MapReduce-style SQL aggregation (GROUP BY + ORDER BY).
    """
    conn  = get_db()
    rows  = conn.execute(
        """
        SELECT
            m.movie_id,
            m.title,
            m.genres,
            COUNT(r.rating)       AS rating_count,
            ROUND(AVG(r.rating), 2) AS avg_rating
        FROM movies m
        JOIN ratings r USING (movie_id)
        GROUP BY m.movie_id
        ORDER BY rating_count DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── /stats/user ───────────────────────────────────────────────────────────────

@app.get("/stats/user/{user_id}", tags=["Analytics"])
def user_stats(user_id: int):
    """Returns a rating summary for a specific user."""
    conn = get_db()
    row  = conn.execute(
        """
        SELECT
            COUNT(*)                AS total_ratings,
            ROUND(AVG(rating), 2)   AS avg_rating,
            MAX(rating)             AS max_rating,
            MIN(rating)             AS min_rating,
            COUNT(DISTINCT movie_id) AS unique_movies
        FROM ratings
        WHERE user_id = ?
        """,
        (user_id,),
    ).fetchone()
    conn.close()
    if not row or row["total_ratings"] == 0:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")
    return {"user_id": user_id, **dict(row)}
