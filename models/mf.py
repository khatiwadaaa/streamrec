"""
models/mf.py
Matrix factorization via TruncatedSVD (scikit-learn).
"""

import os
import pickle
import sqlite3

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

DB_PATH      = "data/streamrec.db"
MODEL_CACHE  = "data/mf_model.pkl"
N_COMPONENTS = 50


def train(db_path: str = DB_PATH, cache_path: str = MODEL_CACHE) -> dict:
    print("Loading ratings …")
    conn   = sqlite3.connect(db_path)
    df     = pd.read_sql("SELECT user_id, movie_id, rating FROM ratings", conn)
    conn.close()

    matrix = df.pivot_table(
        index="user_id", columns="movie_id", values="rating", fill_value=0
    )
    user_ids  = list(matrix.index)
    movie_ids = list(matrix.columns)
    mat       = matrix.values.astype(np.float32)

    print(f"Training SVD ({N_COMPONENTS} factors) on {mat.shape} matrix …")
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
    U   = svd.fit_transform(mat)
    Vt  = svd.components_

    payload = {"U": U, "Vt": Vt, "user_ids": user_ids, "movie_ids": movie_ids, "mat": mat}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Model saved → {cache_path}")
    return payload


def load_model(cache_path: str = MODEL_CACHE) -> dict:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"No model cache at {cache_path}. Run: python models/mf.py")
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def recommend_mf(
    user_id: int,
    n: int = 10,
    model: dict | None = None,
    cache_path: str = MODEL_CACHE,
    force_include_id: int | None = None,  # item to NOT mask (eval use)
) -> list[dict]:
    if model is None:
        model = load_model(cache_path)

    U, Vt     = model["U"], model["Vt"]
    user_ids  = model["user_ids"]
    movie_ids = model["movie_ids"]
    mat       = model["mat"]

    if user_id not in user_ids:
        return []

    user_idx = user_ids.index(user_id)
    user_vec = U[user_idx]
    scores   = user_vec @ Vt

    already_rated = mat[user_idx] > 0
    if force_include_id is not None and force_include_id in movie_ids:
        already_rated[movie_ids.index(force_include_id)] = False

    scores[already_rated] = -np.inf

    top_n_idxs = np.argsort(scores)[-n:][::-1]
    return [
        {"movie_id": int(movie_ids[i]), "score": float(scores[i]), "model": "mf"}
        for i in top_n_idxs
        if np.isfinite(scores[i])
    ]


if __name__ == "__main__":
    train()
