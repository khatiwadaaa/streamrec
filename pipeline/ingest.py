"""
pipeline/ingest.py
Downloads MovieLens 1M, parses it, and loads it into SQLite.
Run once to seed the DB, or schedule it to refresh.
"""

import io
import os
import sqlite3
import urllib.request
import zipfile

import pandas as pd

DB_PATH  = os.environ.get("DB_PATH", "data/streamrec.db")
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


# ── helpers ──────────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")   # safe concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    with open("data/schema.sql") as f:
        conn.executescript(f.read())
    conn.commit()


# ── download & parse ──────────────────────────────────────────────────────────

def fetch_movielens() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download the zip and return (ratings_df, movies_df)."""
    print("Downloading MovieLens 1M …")
    with urllib.request.urlopen(DATA_URL) as resp:
        raw = resp.read()

    zf = zipfile.ZipFile(io.BytesIO(raw))

    # ratings.dat  →  UserID::MovieID::Rating::Timestamp
    ratings_bytes = zf.read("ml-1m/ratings.dat")
    ratings = pd.read_csv(
        io.BytesIO(ratings_bytes),
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )

    # movies.dat  →  MovieID::Title::Genres
    movies_bytes = zf.read("ml-1m/movies.dat")
    movies = pd.read_csv(
        io.BytesIO(movies_bytes),
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )

    return ratings, movies


# ── clean ─────────────────────────────────────────────────────────────────────

def clean_ratings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["user_id", "movie_id"])
    df = df.dropna()
    df["rating"]    = df["rating"].astype(float)
    df["timestamp"] = df["timestamp"].astype(int)
    return df


def clean_movies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["movie_id"])
    df = df.dropna()
    return df


# ── load ──────────────────────────────────────────────────────────────────────

def load_to_db(
    conn: sqlite3.Connection,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> None:
    print(f"Writing {len(movies):,} movies …")
    movies.to_sql("movies", conn, if_exists="replace", index=False)

    print(f"Writing {len(ratings):,} ratings …")
    ratings.to_sql("ratings", conn, if_exists="replace", index=False)

    conn.commit()
    print("Done.")


# ── quick stats ───────────────────────────────────────────────────────────────

def print_stats(conn: sqlite3.Connection) -> None:
    cur = conn.execute(
        """
        SELECT
            COUNT(*)                              AS total_ratings,
            COUNT(DISTINCT user_id)               AS unique_users,
            COUNT(DISTINCT movie_id)              AS unique_movies,
            ROUND(AVG(rating), 2)                 AS avg_rating,
            COUNT(*) * 100.0
              / (COUNT(DISTINCT user_id)
                 * COUNT(DISTINCT movie_id))      AS density_pct
        FROM ratings
        """
    )
    row = cur.fetchone()
    labels = ["total ratings", "unique users", "unique movies",
              "avg rating", "density %"]
    for label, val in zip(labels, row):
        print(f"  {label:<20} {val}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    conn = get_connection()
    init_schema(conn)

    ratings, movies = fetch_movielens()
    ratings = clean_ratings(ratings)
    movies  = clean_movies(movies)

    load_to_db(conn, ratings, movies)
    print("\nDataset stats:")
    print_stats(conn)
    conn.close()
