"""
models/cf.py
User-based collaborative filtering using cosine similarity.
"""

import sqlite3
import numpy as np
import pandas as pd

DB_PATH = "data/streamrec.db"


def load_matrix(conn: sqlite3.Connection):
    df = pd.read_sql("SELECT user_id, movie_id, rating FROM ratings", conn)
    matrix = df.pivot_table(
        index="user_id", columns="movie_id", values="rating", fill_value=0
    )
    return matrix, list(matrix.index), list(matrix.columns)


def cosine_similarity_row(matrix: np.ndarray, user_idx: int) -> np.ndarray:
    user_vec  = matrix[user_idx]
    norms     = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1e-9
    user_norm = np.linalg.norm(user_vec) or 1e-9
    return (matrix @ user_vec) / (norms * user_norm)


def recommend_cf(
    user_id: int,
    n: int = 10,
    k_neighbors: int = 20,
    db_path: str = DB_PATH,
    exclude_ids: set | None = None,   # items to mask (normal use)
    force_include_id: int | None = None,  # item to NOT mask (eval use)
) -> list[dict]:
    conn   = sqlite3.connect(db_path)
    matrix, user_ids, movie_ids = load_matrix(conn)
    conn.close()

    if user_id not in user_ids:
        return []

    mat      = matrix.values.astype(np.float32)
    user_idx = user_ids.index(user_id)

    sims = cosine_similarity_row(mat, user_idx)
    sims[user_idx] = -1

    neighbor_idxs    = np.argsort(sims)[-k_neighbors:][::-1]
    neighbor_sims    = sims[neighbor_idxs].reshape(-1, 1)
    neighbor_ratings = mat[neighbor_idxs]
    scores           = (neighbor_sims * neighbor_ratings).sum(axis=0)

    # mask already-rated — but optionally unblock one item for eval
    already_rated = mat[user_idx] > 0
    if force_include_id is not None and force_include_id in movie_ids:
        already_rated[movie_ids.index(force_include_id)] = False

    scores[already_rated] = -1

    if exclude_ids:
        for mid in exclude_ids:
            if mid in movie_ids:
                scores[movie_ids.index(mid)] = -1

    top_n_idxs = np.argsort(scores)[-n:][::-1]
    return [
        {"movie_id": int(movie_ids[i]), "score": float(scores[i]), "model": "cf"}
        for i in top_n_idxs
        if scores[i] > 0
    ]
