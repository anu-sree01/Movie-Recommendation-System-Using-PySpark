"""Bounded parallel, one-shot OMDb lookups for currently rendered movie cards."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import pandas as pd
import requests

from scripts.fetch_media import DEFAULT_LINKS, imdb_id, poster_fields, read_env_key
from . import database as db
from .omdb_matching import match_omdb_record

_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="cinesphere-omdb")
_LINKS: dict[int, dict] | None = None


def _links_index() -> dict[int, dict]:
    global _LINKS
    if _LINKS is None:
        _LINKS = {}
        if DEFAULT_LINKS.is_file():
            frame = pd.read_csv(DEFAULT_LINKS, dtype={"imdbId": "string"})
            if {"movieId", "imdbId"}.issubset(frame.columns):
                frame["movieId"] = pd.to_numeric(frame["movieId"], errors="coerce")
                frame = frame.dropna(subset=["movieId"]).drop_duplicates("movieId")
                _LINKS = frame.set_index(frame.movieId.astype(int)).to_dict("index")
    return _LINKS


def _fetch_and_store(movie: dict, linked_imdb: str, key: str) -> dict:
    movie_id = int(movie["movieId"])
    session = requests.Session()
    session.headers.update({"User-Agent": "CineSphere academic media reference checker"})
    result = poster_fields(session, api_key=key, linked_imdb=linked_imdb, year=movie.get("year"), timeout=4)
    status = result.get("poster_status", "error")
    if status == "verified":
        accepted, low_confidence, score, reason = match_omdb_record(
            str(movie.get("title", "")), movie.get("year"), result.get("omdb_title", ""), result.get("omdb_year", "")
        )
        if not accepted:
            status = "mismatch_review"
        else:
            status = "ok"
        result.update({"low_confidence": low_confidence, "title_similarity": score, "match_reason": reason})
    elif status == "not_available":
        status = "poster_na"
    elif status == "not_found":
        status = "not_found"
    elif status in {"daily_limit", "invalid_key"}:
        status = "error"
    else:
        status = "error"
    db.save_movie_media_cache(movie_id, result.get("poster_url", "") if status == "ok" else "",
                              result.get("overview", ""), status)
    return {
        "movieId": movie_id, "poster_url": result.get("poster_url", "") if status == "ok" else "",
        "overview": result.get("overview", ""), "poster_status": status,
        "poster_source": "OMDb (IMDb ID matched)", "low_confidence": result.get("low_confidence", False),
    }


def fetch_visible_movies(movies, media: dict, *, max_wait: float = 4.3,
                         max_requests: int = 48, daily_limit: int = 950) -> int:
    """Queue unseen rendered movies on an eight-worker pool; wait only for the page budget."""
    key = read_env_key(Path(__file__).resolve().parents[2] / ".env")
    if not key:
        return 0
    links = _links_index()
    selected = []
    for movie in movies:
        try:
            movie_id = int(movie["movieId"])
        except (KeyError, TypeError, ValueError):
            continue
        asset = media.get(movie_id, {})
        if asset and (asset.get("poster_status") or asset.get("poster_url") or asset.get("trailer_status") or asset.get("trailer_url")):
            continue
        cached = db.get_movie_media_cache(movie_id)
        if cached:
            media[movie_id] = {"movieId": movie_id, "poster_url": cached.get("poster_url", ""),
                               "overview": cached.get("plot_text", ""), "poster_status": cached.get("status", ""),
                               "poster_source": "OMDb (IMDb ID matched)"}
            continue
        linked = imdb_id(links.get(movie_id, {}).get("imdbId", ""))
        if not linked:
            db.save_movie_media_cache(movie_id, "", "", "no_imdb_id")
            continue
        if db.claim_movie_media_with_budget(movie_id, daily_limit=daily_limit):
            selected.append((dict(movie), linked))
        if len(selected) >= max_requests:
            break
    futures = [_POOL.submit(_fetch_and_store, movie, linked, key) for movie, linked in selected]
    if not futures:
        return 0
    # Callers wrap this in st.spinner; wait is bounded independently of queue size.
    done, _ = wait(futures, timeout=max_wait)
    for future in done:
        try:
            row = future.result()
        except Exception:
            continue
        media[int(row["movieId"])] = row
    return len(futures)
