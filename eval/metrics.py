"""
eval/metrics.py
Leave-one-out evaluation: holds out each user's top-rated movie,
asks the model to recommend it back, measures Hit Rate@K and NDCG@K.

Usage:
    python eval/metrics.py
"""

import sqlite3
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from models.cf import recommend_cf
from models.mf import load_model, recommend_mf

DB_PATH   = "data/streamrec.db"
K         = 10
N_USERS   = 300
MIN_RATED = 20


def load_holdout(conn: sqlite3.Connection) -> dict[int, int]:
    """Hold out each user's single highest-rated movie."""
    df  = pd.read_sql("SELECT user_id, movie_id, rating FROM ratings", conn)
    idx = df.groupby("user_id")["rating"].idxmax()
    return df.loc[idx].set_index("user_id")["movie_id"].astype(int).to_dict()


def hit_at_k(recommended: list[int], target: int, k: int) -> float:
    return 1.0 if target in recommended[:k] else 0.0


def ndcg_at_k(recommended: list[int], target: int, k: int) -> float:
    if target not in recommended[:k]:
        return 0.0
    rank = recommended[:k].index(target)
    return 1.0 / math.log2(rank + 2)


def evaluate():
    conn    = sqlite3.connect(DB_PATH)
    holdout = load_holdout(conn)
    counts  = pd.read_sql(
        "SELECT user_id, COUNT(*) as cnt FROM ratings GROUP BY user_id", conn
    )
    conn.close()

    eligible   = set(counts[counts["cnt"] >= MIN_RATED]["user_id"])
    eval_users = [uid for uid in holdout if uid in eligible][:N_USERS]

    print(f"Evaluating on {len(eval_users)} users (K={K}, leave-one-out) …\n")

    mf_model = None
    try:
        mf_model = load_model()
    except FileNotFoundError:
        print("WARNING: MF model not found. Run `python models/mf.py` first.\n")

    cf_hits, cf_ndcg_scores = [], []
    mf_hits, mf_ndcg_scores = [], []

    for i, uid in enumerate(eval_users):
        if i % 50 == 0:
            print(f"  {i}/{len(eval_users)} users done …")

        target = holdout[uid]

        # pass force_include_id so the model doesn't mask the held-out item
        cf_recs = [
            c["movie_id"] for c in
            recommend_cf(uid, n=K * 5, force_include_id=target)
        ]
        if cf_recs:
            cf_hits.append(hit_at_k(cf_recs, target, K))
            cf_ndcg_scores.append(ndcg_at_k(cf_recs, target, K))

        if mf_model:
            mf_recs = [
                c["movie_id"] for c in
                recommend_mf(uid, n=K * 5, model=mf_model, force_include_id=target)
            ]
            if mf_recs:
                mf_hits.append(hit_at_k(mf_recs, target, K))
                mf_ndcg_scores.append(ndcg_at_k(mf_recs, target, K))

    def fmt(lst):
        return f"{np.mean(lst):.4f}" if lst else "  N/A  "

    print("\n" + "─" * 46)
    print(f"{'Model':<12}  {'Hit Rate@'+str(K):<16}  {'NDCG@'+str(K):<12}")
    print("─" * 46)
    print(f"{'CF':<12}  {fmt(cf_hits):<16}  {fmt(cf_ndcg_scores):<12}")
    if mf_model:
        print(f"{'MF (SVD)':<12}  {fmt(mf_hits):<16}  {fmt(mf_ndcg_scores):<12}")
    print("─" * 46)
    print(f"\nUsers evaluated : {len(eval_users)}")
    print(f"K               : {K}")
    print(f"Method          : leave-one-out (held-out = user's top-rated movie)")
    print(f"CF users scored : {len(cf_hits)}")
    if mf_model:
        print(f"MF users scored : {len(mf_hits)}")


if __name__ == "__main__":
    evaluate()
