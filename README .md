# StreamRec — Content Recommendation Engine

Full recommendation pipeline: data ingestion → candidate generation (CF + MF) → weighted re-ranking → FastAPI serving layer → Go in-memory cache.

Built on the MovieLens 1M dataset — 1,000,209 ratings, 6,040 users, 3,706 movies.

---

## Architecture

```
Browser / curl
      │
      │  HTTP REST
      ▼
FastAPI server — port 8000          (api/server.py)
      │
      ├── GET /recommend ──► checks Go cache first
      │         │
      │         └── cache miss → compute
      │               ├── CF model    (models/cf.py)      cosine similarity
      │               ├── MF model    (models/mf.py)      SVD latent factors
      │               └── Ranker      (models/ranker.py)  weighted blend
      │
      ├── POST /feedback ──► SQLite ab_log table (A/B logging)
      ├── GET  /stats/* ───► SQL aggregation queries
      └── GET  /health ────► SRE uptime check

Go cache — port 8080                (go-cache/main.go)
      LRU cache, cap=1,000, TTL=5min
      Hit → returns cached JSON directly, skips Python compute

Data pipeline                       (pipeline/ingest.py)
      Download MovieLens 1M → clean with Pandas → load into SQLite
      Tables: ratings, movies, ab_log
```

---

## Evaluation results

Evaluated on 300 users via leave-one-out: hold out each user's top-rated movie, ask the model for top-10 recommendations, check if the held-out movie appears.

| Model    | Hit Rate@10 | NDCG@10 |
|----------|-------------|---------|
| CF       | 0.4700      | 0.3201  |
| MF (SVD) | 0.5300      | 0.3711  |

*Hit Rate@10 = fraction of users whose top-rated movie appeared in the model's top-10 recommendations across a catalog of 3,706 movies.*

---

## API endpoints

| Method | Endpoint                | Description                         |
|--------|-------------------------|-------------------------------------|
| GET    | `/health`               | SRE uptime check                    |
| GET    | `/recommend`            | Top-N recommendations for a user    |
| POST   | `/feedback`             | Log click / skip / watch event      |
| GET    | `/stats/top-movies`     | Most-rated movies (SQL aggregation) |
| GET    | `/stats/user/{user_id}` | Per-user rating summary             |

**GET /recommend params:**
- `user_id` (required) — integer user ID
- `n` (default 10) — number of results, max 50
- `model` (default `hybrid`) — `cf` | `mf` | `hybrid`

**Example:**
```bash
curl "http://localhost:8000/recommend?user_id=1&n=10&model=hybrid"
```

```json
{
  "user_id": 1,
  "model": "hybrid",
  "count": 10,
  "results": [
    {
      "movie_id": 2081,
      "final_score": 0.7869,
      "cf_score": 1.0,
      "mf_score": 0.6816,
      "pop_score": 0.3019,
      "title": "Little Mermaid, The (1989)",
      "genres": "Animation|Children's|Comedy|Musical|Romance"
    }
  ]
}
```

---

## Setup

**Prerequisites:** Python 3.11+, Go 1.21+

```bash
# 1. Clone
git clone https://github.com/khatiwadaaa/streamrec
cd streamrec

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download and ingest MovieLens 1M into SQLite (~2 min)
python pipeline/ingest.py

# 4. Train the MF model and cache to disk (~60s)
python models/mf.py

# 5. Start the Go cache (terminal 1)
cd go-cache && go run main.go

# 6. Start the FastAPI server (terminal 2)
uvicorn api.server:app --reload

# 7. Open interactive API docs
open http://localhost:8000/docs
```

---

## How it works

**Collaborative filtering (CF)** finds users with similar rating histories using cosine similarity, then aggregates their top-rated unseen movies as candidates. Pure NumPy — no ML framework.

**Matrix factorization (MF)** decomposes the full 6,040×3,706 user–item matrix into 50 latent factors using SVD. A user's score for an unseen movie is the dot product of their latent vector against the movie's latent vector.

**Re-ranking** merges CF and MF candidates, normalizes scores to [0,1], then blends: `score = 0.45·CF + 0.45·MF + 0.10·popularity`. Weights are configurable for A/B testing.

**Go LRU cache** sits in front of the Python compute layer with cap=1,000 and TTL=5min. Repeat requests for the same user/model are served in microseconds instead of recomputing.

**A/B logging** persists every click, skip, and watch event to SQLite for offline analysis via `POST /feedback`.

---

## Project structure

```
streamrec/
├── pipeline/ingest.py      # download, clean, load MovieLens into SQLite
├── models/cf.py            # user-based collaborative filtering
├── models/mf.py            # SVD matrix factorization
├── models/ranker.py        # weighted re-ranking layer
├── api/server.py           # FastAPI endpoints + Go cache integration
├── eval/metrics.py         # Hit Rate@K and NDCG@K evaluation
├── go-cache/main.go        # LRU cache microservice (Go)
├── data/schema.sql         # SQLite schema — ratings, movies, ab_log
└── requirements.txt
```

---

## Skills demonstrated

Python · Pandas · NumPy · SciPy · scikit-learn · SQLite · FastAPI · REST API design · collaborative filtering · matrix factorization · recommendation systems · data pipelines · MapReduce-style aggregation · Go (Golang) · LRU cache · A/B logging · SRE · client-server architecture · OOP
