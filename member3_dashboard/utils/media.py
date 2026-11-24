"""Read verified MovieLens movieId-to-TMDB media metadata."""

import json
from pathlib import Path

import pandas as pd

from . import database as db

MEDIA_DIR = Path(__file__).resolve().parents[1] / "media"
METADATA_FILE = MEDIA_DIR / "metadata" / "movie_media.csv"
LEGACY_METADATA_FILE = MEDIA_DIR / "metadata" / "tmdb_media.json"


def load_media(path: Path = METADATA_FILE) -> dict[int, dict]:
    records = []
    if path.is_file():
        try:
            frame = pd.read_csv(path, low_memory=False)
            records = frame.astype(object).where(pd.notna(frame), None).to_dict("records")
        except (OSError, pd.errors.ParserError, UnicodeError):
            records = []
    if not records and LEGACY_METADATA_FILE.is_file():
        try:
            records = json.loads(LEGACY_METADATA_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            records = []
    media = {}
    for record in records:
        try:
            item = dict(record)
            if item.get("poster_path") and not item.get("poster_url"):
                local_path = (MEDIA_DIR / str(item["poster_path"])).resolve()
                if local_path.is_file() and local_path.is_relative_to(MEDIA_DIR.resolve()):
                    item["poster_url"] = str(local_path)
                    item.setdefault("poster_status", "verified")
            media[int(item["movieId"])] = item
        except (KeyError, TypeError, ValueError):
            continue
    # The durable batch manifest wins; on-demand results fill only missing IDs.
    try:
        with db.connect() as conn:
            cache_records = conn.execute(
                "SELECT movie_id,poster_url,plot_text,status,fetched_at FROM movie_media_cache"
            ).fetchall()
        for row in cache_records:
            movie_id = int(row["movie_id"])
            manifest_item = media.get(movie_id, {})
            manifest_status = str(manifest_item.get("poster_status") or "").strip().lower()
            if movie_id not in media or not manifest_status:
                media[movie_id] = {
                    "movieId": movie_id,
                    "poster_url": row["poster_url"],
                    "overview": row["plot_text"],
                    "poster_status": row["status"],
                    "fetched_at": row["fetched_at"],
                    "poster_source": "OMDb (IMDb ID matched)",
                }
    except Exception:
        # Loading the catalogue remains available if an older/read-only database
        # cannot yet provide the optional on-demand cache table.
        pass
    return media
