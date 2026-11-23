"""Optionally acquire TMDB poster and YouTube trailer references.

The dashboard also accepts manually curated, source-reviewed rows in
media/metadata/movie_media.csv; this TMDB adapter is not required by the app.
A token is needed only for batches that contain TMDB IDs. This tool stores
references and statuses; it does not download or rehost media files.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from member3_dashboard.utils.catalog import MOVIES_FILE


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LINKS = ROOT / "data" / "raw" / "links.csv"
MEDIA_DIR = ROOT / "member3_dashboard" / "media" / "metadata"
CACHE_FILE = MEDIA_DIR / "tmdb_media_cache.json"
METADATA_FILE = MEDIA_DIR / "movie_media.csv"
REPORT_FILE = MEDIA_DIR / "media_acquisition_report.json"
PROGRESS_FILE = MEDIA_DIR / "acquisition_progress.json"
TMDB_API = "https://api.themoviedb.org/3/movie/{}"
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


def find_links(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit else [
        DEFAULT_LINKS,
        ROOT / "data" / "raw" / "ml-32m" / "links.csv",
        ROOT / "dataset" / "ml-32m" / "links.csv",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    raise FileNotFoundError("MovieLens links.csv was not found. Place it in data/raw/ or pass --links.")


def request_with_retry(session: requests.Session, url: str, *, headers=None, params=None, attempts=4):
    last_error = "request failed"
    for attempt in range(attempts):
        try:
            response = session.get(url, headers=headers, params=params, timeout=(10, 25), stream=False)
            if response.status_code == 429 or response.status_code >= 500:
                delay = float(response.headers.get("Retry-After", min(2 ** attempt, 30)))
                last_error = f"HTTP {response.status_code}"
                response.close()
                if attempt + 1 < attempts:
                    time.sleep(delay)
                    continue
            elif 400 <= response.status_code < 500:
                return None, f"HTTP {response.status_code}: {response.text[:300]}"
            response.raise_for_status()
            return response, None
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 30))
    return None, last_error


def _valid_poster(session: requests.Session, poster_path: str) -> tuple[str | None, str | None]:
    url = IMAGE_BASE + poster_path
    response, error = request_with_retry(session, url)
    if error:
        return None, f"poster validation failed: {error}"
    content_type = response.headers.get("Content-Type", "")
    response.close()
    if not content_type.startswith("image/"):
        return None, f"poster validation failed: unexpected content type {content_type or 'unknown'}"
    return url, None


def _valid_trailer(session: requests.Session, video: dict) -> tuple[str | None, str | None]:
    key = str(video.get("key", "")).strip()
    url = f"https://www.youtube.com/watch?v={key}"
    response, error = request_with_retry(session, "https://www.youtube.com/oembed", params={"url": url, "format": "json"})
    if error:
        return None, f"trailer validation failed: {error}"
    response.close()
    return url, None


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json_atomic(path: Path, payload: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _write_manifest(cache: dict[int, dict], catalog_count: int):
    old_rows = []
    if METADATA_FILE.is_file():
        try:
            old_frame = pd.read_csv(METADATA_FILE)
            old_rows = old_frame.astype(object).where(pd.notna(old_frame), None).to_dict("records")
        except (OSError, pd.errors.ParserError, UnicodeError):
            old_rows = []
    rows = {}
    for row in old_rows:
        movie_id = pd.to_numeric(row.get("movieId"), errors="coerce")
        if pd.notna(movie_id):
            rows[int(movie_id)] = dict(row)
    for movie_id, item in cache.items():
        merged = rows.setdefault(int(movie_id), {})
        for key, value in item.items():
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            # Preserve human-reviewed candidates unless an API lookup yields a verified asset.
            if key.startswith("poster_") and merged.get("poster_status") == "permission_required" and item.get("poster_status") != "verified":
                continue
            if key.startswith("trailer_") and merged.get("trailer_status") in {"candidate_unverified", "search_only"} and item.get("trailer_status") != "verified":
                continue
            merged[key] = value
        if item.get("poster_status") == "verified":
            merged["poster_source"] = f"TMDB movie {item.get('tmdbId')} image reference"
            merged["licensing_status"] = "TMDB remote image terms apply; local reuse not assessed"
            merged["verification_status"] = "Poster URL responded with image content type; MovieLens links.csv tmdbId matched the API response ID."
        if item.get("trailer_status") == "verified":
            merged["trailer_source"] = "TMDB video metadata and YouTube oEmbed"
            merged["trailer_channel"] = "YouTube; TMDB marked video official"
            merged["verification_status"] = "YouTube oEmbed responded and TMDB marked the trailer official; playback was not automated."
        merged["movieId"] = int(movie_id)
    columns = [
        "movieId", "title", "poster_path", "poster_url", "poster_source", "poster_status",
        "licensing_status", "trailer_url", "trailer_channel", "trailer_source", "trailer_status",
        "verification_status", "overview", "poster_error", "trailer_error", "tmdbId", "attempted_at", "source", "status",
    ]
    frame = pd.DataFrame(rows.values())
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    frame[columns].sort_values("movieId").to_csv(METADATA_FILE, index=False)

    all_rows = frame[columns].to_dict("records")
    previous_report = _read_json(REPORT_FILE, {})
    def local_poster_exists(value):
        if value is None or pd.isna(value) or not str(value).strip():
            return False
        candidate = (ROOT / "member3_dashboard" / "media" / str(value)).resolve()
        return candidate.is_relative_to((ROOT / "member3_dashboard" / "media").resolve()) and candidate.is_file()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "catalog_movies": catalog_count,
        "movies_with_metadata": len(all_rows),
        "movies_not_yet_inspected": max(0, catalog_count - len(all_rows)),
        "movies_web_searched": len(_read_json(PROGRESS_FILE, {}).get("searched_movie_ids", [])),
        "poster_files_obtained": sum(local_poster_exists(row.get("poster_path")) for row in all_rows),
        "verified_remote_poster_references": sum(row.get("poster_status") == "verified" and bool(row.get("poster_url")) for row in all_rows),
        "posters_with_permission_required": sum(row.get("poster_status") == "permission_required" for row in all_rows),
        "verified_official_trailers": sum(row.get("trailer_status") == "verified" for row in all_rows),
        "unverified_trailer_references": sum(row.get("trailer_status") == "candidate_unverified" for row in all_rows),
        "trailer_search_links_only": sum(row.get("trailer_status") == "search_only" for row in all_rows),
        "movies_without_suitable_trailers": sum(row.get("trailer_status") == "unavailable_confirmed" for row in all_rows),
        "failed_media_attempts": sum(row.get("poster_status") == "failed" or row.get("trailer_status") == "failed" for row in all_rows),
        "sources_used": list(dict.fromkeys([*(previous_report.get("sources_used", [])), "TMDB API and image CDN", "YouTube oEmbed"])),
        "limitations": ["A metadata match or search result is not a substitute for checking licensing and actual playback.", "No images are downloaded by this tool; it validates references and records source/status."],
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def acquire(links_path: Path, *, offset=0, limit=100, retry_failed=False, force=False) -> list[dict]:
    movies = pd.read_csv(MOVIES_FILE, usecols=["movieId", "title", "year"])
    links = pd.read_csv(links_path, usecols=["movieId", "tmdbId"])
    links["movieId"] = pd.to_numeric(links["movieId"], errors="coerce")
    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
    catalog_ids = set(pd.to_numeric(movies["movieId"], errors="coerce").dropna().astype("int64"))
    links = links.dropna(subset=["movieId"]).copy()
    links["movieId"] = links["movieId"].astype("int64")
    links = links.loc[links["movieId"].isin(catalog_ids)].drop_duplicates("movieId")
    catalog_links = movies.copy()
    catalog_links["movieId"] = pd.to_numeric(catalog_links["movieId"], errors="coerce").astype("int64")
    catalog_links = catalog_links.merge(links, on="movieId", how="left").sort_values("movieId")
    batch = catalog_links.iloc[offset:offset + limit]

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    previous = _read_json(CACHE_FILE, [])
    cache = {int(row["movieId"]): row for row in previous if isinstance(row, dict) and str(row.get("movieId", "")).isdigit()}
    progress = _read_json(PROGRESS_FILE, {})
    if progress.get("active_offset") == offset:
        partial = progress.get("in_progress_records", [])
        cache.update({int(row["movieId"]): row for row in partial if str(row.get("movieId", "")).isdigit()})
    else:
        progress["active_offset"] = offset
        progress["in_progress_records"] = []

    token = os.environ.get("TMDB_READ_ACCESS_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    api_key = os.environ.get("TMDB_API_KEY")
    params = {} if token or not api_key else {"api_key": api_key}
    has_mapped_id = pd.to_numeric(batch["tmdbId"], errors="coerce").gt(0).any()
    if has_mapped_id and not token and not api_key:
        raise RuntimeError("This batch includes TMDB IDs. To use this optional metadata source, set TMDB_READ_ACCESS_TOKEN or TMDB_API_KEY. The dashboard itself does not require either.")
    session = requests.Session()

    def save_partial():
        progress["catalog_movies"] = len(catalog_ids)
        progress["active_offset"] = offset
        progress["in_progress_records"] = [cache[int(mid)] for mid in batch["movieId"].tolist() if int(mid) in cache]
        _write_json_atomic(PROGRESS_FILE, progress)

    for link in batch.itertuples(index=False):
        movie_id = int(link.movieId)
        old = cache.get(movie_id, {})
        if not force and old and (not retry_failed or old.get("status") == "unavailable"):
            continue
        if pd.isna(link.tmdbId) or int(link.tmdbId) <= 0:
            cache[movie_id] = {
                "movieId": movie_id,
                "title": str(link.title),
                "year": int(link.year) if pd.notna(link.year) else None,
                "tmdbId": None,
                "poster_path": None,
                "poster_url": None,
                "poster_status": "no_tmdb_id",
                "poster_error": "MovieLens links.csv has no TMDB ID for this movie.",
                "trailer_url": None,
                "trailer_channel": None,
                "trailer_status": "no_tmdb_id",
                "trailer_error": "MovieLens links.csv has no TMDB ID for this movie.",
                "status": "unavailable",
                "attempted_at": datetime.now(timezone.utc).isoformat(),
                "source": "TMDB API",
            }
            save_partial()
            continue
        tmdb_id = int(link.tmdbId)

        record = {
            "movieId": movie_id,
            "title": str(link.title),
            "year": int(link.year) if pd.notna(link.year) else None,
            "tmdbId": tmdb_id,
            "poster_path": None,
            "poster_url": None,
            "poster_status": "missing",
            "poster_error": None,
            "trailer_url": None,
            "trailer_channel": None,
            "trailer_status": "missing",
            "trailer_error": None,
            "attempted_at": datetime.now(timezone.utc).isoformat(),
            "source": "TMDB API",
        }
        detail_params = {**params, "append_to_response": "videos"}
        response, error = request_with_retry(session, TMDB_API.format(tmdb_id), headers=headers, params=detail_params)
        if error:
            record["poster_status"] = record["trailer_status"] = "failed"
            record["poster_error"] = record["trailer_error"] = f"TMDB metadata request failed: {error}"
        else:
            try:
                data = response.json()
            except requests.JSONDecodeError as exc:
                data = None
                record["poster_status"] = record["trailer_status"] = "failed"
                record["poster_error"] = record["trailer_error"] = f"TMDB returned invalid JSON: {exc}"
            response.close()
            if data is not None and not isinstance(data, dict):
                record["poster_status"] = record["trailer_status"] = "failed"
                record["poster_error"] = record["trailer_error"] = "TMDB returned an unexpected response shape."
                data = None
            if data is not None:
                if int(data.get("id", 0) or 0) != tmdb_id:
                    record["poster_status"] = record["trailer_status"] = "failed"
                    mismatch = f"TMDB response ID did not match the links.csv tmdbId {tmdb_id}."
                    record["poster_error"] = record["trailer_error"] = mismatch
                    data = None
            if data is not None:
                poster_path = data.get("poster_path")
                record["overview"] = data.get("overview") or None
                if poster_path:
                    poster_url, poster_error = _valid_poster(session, poster_path)
                    if poster_url:
                        record["poster_path"], record["poster_url"], record["poster_status"] = poster_path, poster_url, "verified"
                    else:
                        record["poster_status"] = "failed"
                        record["poster_error"] = poster_error
                else:
                    record["poster_status"] = "not_available"
                    record["poster_error"] = "TMDB returned no poster_path for this title."

                videos = data.get("videos", {}).get("results", [])
                candidates = [v for v in videos if v.get("site") == "YouTube" and v.get("type") == "Trailer" and v.get("key")]
                candidates.sort(key=lambda v: (v.get("official") is True, v.get("published_at", "")), reverse=True)
                if candidates:
                    trailer_url, trailer_error = _valid_trailer(session, candidates[0])
                    if trailer_url:
                        record["trailer_url"] = trailer_url
                        record["trailer_channel"] = "YouTube; TMDB video metadata"
                        record["trailer_status"] = "verified" if candidates[0].get("official") is True else "candidate_unverified"
                    else:
                        record["trailer_status"] = "failed"
                        record["trailer_error"] = trailer_error
                else:
                    record["trailer_status"] = "not_available"
                    record["trailer_error"] = "TMDB returned no official YouTube trailer for this title."
        record["status"] = "success" if record["poster_status"] == "verified" or record["trailer_status"] == "verified" else "failed"
        cache[movie_id] = record
        save_partial()
        time.sleep(0.25)

    CACHE_FILE.write_text(json.dumps(list(cache.values()), indent=2, ensure_ascii=False), encoding="utf-8")
    catalog_count = len(catalog_ids)
    _write_manifest(cache, catalog_count)
    progress["last_completed_offset"] = offset
    progress["last_completed_count"] = len(batch)
    progress["in_progress_records"] = []
    _write_json_atomic(PROGRESS_FILE, progress)
    selected_ids = {int(movie_id) for movie_id in batch["movieId"].tolist()}
    return [cache[movie_id] for movie_id in sorted(selected_ids) if movie_id in cache]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links", type=Path, help="MovieLens links.csv; defaults to data/raw/links.csv")
    parser.add_argument("--offset", type=int, default=0, help="Start after this many linked movies")
    parser.add_argument("--limit", type=int, default=100, help="Process at most this many linked movies (default: 100)")
    parser.add_argument("--retry-failed", action="store_true", help="Retry cached unsuccessful records in this batch")
    parser.add_argument("--force", action="store_true", help="Refetch every record in this batch")
    args = parser.parse_args()
    links_path = find_links(args.links)
    records = acquire(links_path, offset=max(0, args.offset), limit=max(1, args.limit), retry_failed=args.retry_failed, force=args.force)
    successful = sum(row.get("status") == "success" for row in records)
    posters = sum(row.get("poster_status") == "verified" for row in records)
    trailers = sum(row.get("trailer_status") == "verified" for row in records)
    no_tmdb = sum(row.get("status") == "unavailable" for row in records)
    print(f"Batch records: {len(records)} | poster references verified: {posters} | official trailers verified: {trailers} | records with media: {successful} | no TMDB ID: {no_tmdb}")
    print(f"Movie media manifest: {METADATA_FILE}")
    print(f"Cumulative coverage report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
