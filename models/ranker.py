"""
models/ranker.py
Merges CF and MF candidate lists and re-ranks with a weighted score.
Configurable weights let you A/B test different blending strategies.
"""

import sqlite3

import numpy as np

DB_PATH = "data/streamrec.db"

# Blending weights — must sum to 1.0
ALPHA = 0.45   # CF weight
BETA  = 0.45   # MF weight
GAMMA = 0.10   # global popularity weight


def _popularity_scores(movie_ids: list[int], conn: sqlite3.Connection) -> dict[int, float]:
    """
    Returns a dict {movie_id: popularity_score} normalized to [0, 1].
    Popularity = number of ratings for that movie.
    """
    placeholders = ",".join("?" * len(movie_ids))
    cur = conn.execute(
        f"""
        SELECT movie_id, COUNT(*) as cnt
        FROM ratings
        WHERE movie_id IN ({placeholders})
        GROUP BY movie_id
        """,
        movie_ids,
    )
    rows  = dict(cur.fetchall())
    if not rows:
        return {}
    max_cnt = max(rows.values()) or 1
    return {mid: cnt / max_cnt for mid, cnt in rows.items()}


def _normalize(candidates: list[dict]) -> dict[int, float]:
    """Min-max normalize scores within a candidate list → {movie_id: score}."""
    if not candidates:
        return {}
    scores = np.array([c["score"] for c in candidates], dtype=np.float32)
    lo, hi = scores.min(), scores.max()
    span   = (hi - lo) or 1.0
    return {c["movie_id"]: float((c["score"] - lo) / span) for c in candidates}


def rank(
    cf_candidates:  list[dict],
    mf_candidates:  list[dict],
    n: int = 10,
    db_path: str = DB_PATH,
    alpha: float = ALPHA,
    beta:  float = BETA,
    gamma: float = GAMMA,
) -> list[dict]:
    """
    Merges and re-ranks CF + MF candidates.

    Args:
        cf_candidates: output of recommend_cf(...)
        mf_candidates: output of recommend_mf(...)
        n:             how many results to return
        db_path:       path to SQLite DB (for popularity lookup)
        alpha/beta/gamma: blending weights

    Returns:
        Sorted list of dicts:
        {"movie_id", "final_score", "cf_score", "mf_score", "pop_score", "model"}
    """
    cf_norm  = _normalize(cf_candidates)
    mf_norm  = _normalize(mf_candidates)
    all_mids = set(cf_norm) | set(mf_norm)

    if not all_mids:
        return []

    conn    = sqlite3.connect(db_path)
    pop     = _popularity_scores(list(all_mids), conn)
    conn.close()

    results = []
    for mid in all_mids:
        cf_s  = cf_norm.get(mid, 0.0)
        mf_s  = mf_norm.get(mid, 0.0)
        pop_s = pop.get(mid, 0.0)
        final = alpha * cf_s + beta * mf_s + gamma * pop_s
        results.append({
            "movie_id":    mid,
            "final_score": round(final, 4),
            "cf_score":    round(cf_s, 4),
            "mf_score":    round(mf_s, 4),
            "pop_score":   round(pop_s, 4),
            "model":       "hybrid",
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:n]
