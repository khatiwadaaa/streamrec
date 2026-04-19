CREATE TABLE IF NOT EXISTS ratings (
    user_id   INTEGER NOT NULL,
    movie_id  INTEGER NOT NULL,
    rating    REAL    NOT NULL,
    timestamp INTEGER NOT NULL,
    PRIMARY KEY (user_id, movie_id)
);

CREATE TABLE IF NOT EXISTS movies (
    movie_id INTEGER PRIMARY KEY,
    title    TEXT    NOT NULL,
    genres   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS ab_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL,
    movie_id   INTEGER NOT NULL,
    event      TEXT    NOT NULL,   -- 'click' | 'skip' | 'watch'
    model      TEXT    NOT NULL,   -- 'cf' | 'mf' | 'hybrid'
    created_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);

CREATE INDEX IF NOT EXISTS idx_ratings_user   ON ratings(user_id);
CREATE INDEX IF NOT EXISTS idx_ratings_movie  ON ratings(movie_id);
CREATE INDEX IF NOT EXISTS idx_ab_log_user    ON ab_log(user_id);
