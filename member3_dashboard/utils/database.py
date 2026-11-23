"""SQLite persistence for application accounts and per-user movie activity."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "database" / "cinesphere.sqlite3"
PROJECT_TIMEZONE = ZoneInfo("Asia/Kolkata")


def omdb_day() -> str:
    return datetime.now(PROJECT_TIMEZONE).date().isoformat()


@contextmanager
def connect(db_path: Path | str = DEFAULT_DB):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize(db_path: Path | str = DEFAULT_DB) -> None:
    with connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                application_user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL CHECK(length(trim(name)) > 0),
                movielens_user_id INTEGER UNIQUE,
                onboarding_complete INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS user_genres (
                application_user_id INTEGER NOT NULL REFERENCES users(application_user_id) ON DELETE CASCADE,
                genre TEXT NOT NULL,
                PRIMARY KEY (application_user_id, genre)
            );
            CREATE TABLE IF NOT EXISTS watchlist (
                application_user_id INTEGER NOT NULL REFERENCES users(application_user_id) ON DELETE CASCADE,
                movieId INTEGER NOT NULL,
                added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (application_user_id, movieId)
            );
            CREATE TABLE IF NOT EXISTS watched_movies (
                application_user_id INTEGER NOT NULL REFERENCES users(application_user_id) ON DELETE CASCADE,
                movieId INTEGER NOT NULL,
                watched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                rating REAL CHECK(rating IS NULL OR (rating >= 0.5 AND rating <= 5.0)),
                PRIMARY KEY (application_user_id, movieId)
            );
            CREATE TABLE IF NOT EXISTS movie_media_cache (
                movie_id INTEGER PRIMARY KEY,
                poster_url TEXT NOT NULL DEFAULT '',
                plot_text TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS omdb_usage (
                day TEXT PRIMARY KEY,
                count INTEGER NOT NULL DEFAULT 0 CHECK(count >= 0)
            );
            """
        )
        user_columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
        if "onboarding_complete" not in user_columns:
            conn.execute("ALTER TABLE users ADD COLUMN onboarding_complete INTEGER NOT NULL DEFAULT 0")


def reserve_omdb_request(daily_limit: int = 950, db_path: Path | str = DEFAULT_DB) -> bool:
    """Atomically reserve one OMDb request shared by the app and batch script."""
    day = omdb_day()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT count FROM omdb_usage WHERE day=?", (day,)).fetchone()
        count = int(row["count"]) if row else 0
        if count >= int(daily_limit):
            return False
        conn.execute(
            "INSERT INTO omdb_usage(day,count) VALUES (?,1) "
            "ON CONFLICT(day) DO UPDATE SET count=count+1",
            (day,),
        )
        return True


def omdb_usage_count(day: str | None = None, db_path: Path | str = DEFAULT_DB) -> int:
    target_day = day or omdb_day()
    with connect(db_path) as conn:
        row = conn.execute("SELECT count FROM omdb_usage WHERE day=?", (target_day,)).fetchone()
        return int(row["count"]) if row else 0


def get_movie_media_cache(movie_id: int, db_path: Path | str = DEFAULT_DB) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM movie_media_cache WHERE movie_id=?", (int(movie_id),)).fetchone()
        return dict(row) if row else None


def movie_media_cache_version(db_path: Path | str = DEFAULT_DB) -> str:
    with connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS n,COALESCE(MAX(fetched_at),'') AS newest FROM movie_media_cache").fetchone()
        statuses = conn.execute("SELECT status,COUNT(*) AS n FROM movie_media_cache GROUP BY status ORDER BY status").fetchall()
        summary = ",".join(f"{item['status']}={item['n']}" for item in statuses)
        return f"{int(row['n'])}:{row['newest']}:{summary}"


def save_movie_media_cache(movie_id: int, poster_url: str, plot_text: str, status: str,
                           db_path: Path | str = DEFAULT_DB) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO movie_media_cache(movie_id,poster_url,plot_text,status,fetched_at) "
            "VALUES (?,?,?,?,CURRENT_TIMESTAMP) ON CONFLICT(movie_id) DO UPDATE SET "
            "poster_url=excluded.poster_url,plot_text=excluded.plot_text,status=excluded.status,fetched_at=excluded.fetched_at",
            (int(movie_id), str(poster_url or ""), str(plot_text or ""), str(status)),
        )


def claim_movie_media(movie_id: int, db_path: Path | str = DEFAULT_DB) -> bool:
    """Persist a one-shot pending marker before a worker makes its HTTP request."""
    with connect(db_path) as conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO movie_media_cache(movie_id,status) VALUES (?, 'pending')",
            (int(movie_id),),
        )
        return cursor.rowcount == 1


def claim_movie_media_with_budget(movie_id: int, daily_limit: int = 950,
                                  db_path: Path | str = DEFAULT_DB) -> bool:
    """Atomically claim a movie and reserve a daily OMDb request."""
    day = omdb_day()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        exists = conn.execute("SELECT 1 FROM movie_media_cache WHERE movie_id=?", (int(movie_id),)).fetchone()
        if exists:
            return False
        usage = conn.execute("SELECT count FROM omdb_usage WHERE day=?", (day,)).fetchone()
        count = int(usage["count"]) if usage else 0
        if count >= int(daily_limit):
            return False
        conn.execute(
            "INSERT INTO omdb_usage(day,count) VALUES (?,1) "
            "ON CONFLICT(day) DO UPDATE SET count=count+1", (day,)
        )
        conn.execute("INSERT INTO movie_media_cache(movie_id,status) VALUES (?, 'pending')", (int(movie_id),))
        return True


def register_user(name: str, genres=(), movielens_user_id: int | None = None, db_path: Path | str = DEFAULT_DB) -> int:
    cleaned_name = name.strip()
    if not cleaned_name:
        raise ValueError("Please enter your name.")
    with connect(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO users(name, movielens_user_id, onboarding_complete) VALUES (?, ?, ?)",
            (cleaned_name, int(movielens_user_id) if movielens_user_id is not None else None, int(bool(genres))),
        )
        user_id = int(cursor.lastrowid)
        conn.executemany(
            "INSERT INTO user_genres(application_user_id, genre) VALUES (?, ?)",
            [(user_id, genre) for genre in sorted(set(genres))],
        )
        return user_id


def list_users(db_path: Path | str = DEFAULT_DB) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY application_user_id").fetchall()
        return [dict(row) for row in rows]


def user_genres(user_id: int, db_path: Path | str = DEFAULT_DB) -> list[str]:
    with connect(db_path) as conn:
        rows = conn.execute("SELECT genre FROM user_genres WHERE application_user_id=? ORDER BY genre", (int(user_id),)).fetchall()
        return [row["genre"] for row in rows]


def save_genres(user_id: int, genres, db_path: Path | str = DEFAULT_DB) -> None:
    with connect(db_path) as conn:
        conn.execute("DELETE FROM user_genres WHERE application_user_id=?", (int(user_id),))
        conn.executemany("INSERT INTO user_genres(application_user_id, genre) VALUES (?, ?)", [(int(user_id), genre) for genre in sorted(set(genres))])
        conn.execute("UPDATE users SET onboarding_complete=1 WHERE application_user_id=?", (int(user_id),))


def set_pilot_profile(user_id: int, movielens_user_id: int | None, db_path: Path | str = DEFAULT_DB) -> None:
    with connect(db_path) as conn:
        conn.execute("UPDATE users SET movielens_user_id=? WHERE application_user_id=?", (int(movielens_user_id) if movielens_user_id is not None else None, int(user_id)))


def watched_ids(user_id: int, db_path: Path | str = DEFAULT_DB) -> set[int]:
    with connect(db_path) as conn:
        return {int(row[0]) for row in conn.execute("SELECT movieId FROM watched_movies WHERE application_user_id=?", (int(user_id),))}


def watchlist_ids(user_id: int, db_path: Path | str = DEFAULT_DB) -> set[int]:
    with connect(db_path) as conn:
        return {int(row[0]) for row in conn.execute("SELECT movieId FROM watchlist WHERE application_user_id=?", (int(user_id),))}


def add_to_watchlist(user_id: int, movie_id: int, db_path: Path | str = DEFAULT_DB) -> None:
    with connect(db_path) as conn:
        conn.execute("INSERT OR IGNORE INTO watchlist(application_user_id, movieId) VALUES (?, ?)", (int(user_id), int(movie_id)))


def remove_from_watchlist(user_id: int, movie_id: int, db_path: Path | str = DEFAULT_DB) -> None:
    with connect(db_path) as conn:
        conn.execute("DELETE FROM watchlist WHERE application_user_id=? AND movieId=?", (int(user_id), int(movie_id)))


def mark_watched(user_id: int, movie_id: int, rating: float | None = None, db_path: Path | str = DEFAULT_DB) -> None:
    if rating is not None and not 0.5 <= float(rating) <= 5.0:
        raise ValueError("Rating must be between 0.5 and 5 stars.")
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO watched_movies(application_user_id, movieId, rating) VALUES (?, ?, ?) "
            "ON CONFLICT(application_user_id, movieId) DO UPDATE SET watched_at=CURRENT_TIMESTAMP, rating=excluded.rating",
            (int(user_id), int(movie_id), float(rating) if rating is not None else None),
        )
        conn.execute("DELETE FROM watchlist WHERE application_user_id=? AND movieId=?", (int(user_id), int(movie_id)))


def remove_watched(user_id: int, movie_id: int, db_path: Path | str = DEFAULT_DB) -> None:
    with connect(db_path) as conn:
        conn.execute("DELETE FROM watched_movies WHERE application_user_id=? AND movieId=?", (int(user_id), int(movie_id)))


def movie_activity(user_id: int, table: str, db_path: Path | str = DEFAULT_DB):
    if table not in {"watchlist", "watched_movies"}:
        raise ValueError("Unsupported activity table.")
    with connect(db_path) as conn:
        return [dict(row) for row in conn.execute(f"SELECT * FROM {table} WHERE application_user_id=? ORDER BY movieId", (int(user_id),))]
