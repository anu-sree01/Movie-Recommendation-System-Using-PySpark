"""Acquire CineSphere poster references in a resumable, popularity-first queue.

Posters are hotlinked from OMDb's response. This script never downloads poster
files or performs per-movie web searches. Statuses and request reservations are
saved so the same daily command can resume after limits or network interruptions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, quote_plus
from datetime import date, timedelta
from math import ceil
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PROJECT_TIMEZONE = ZoneInfo("Asia/Kolkata")


def project_today() -> date:
    return datetime.now(PROJECT_TIMEZONE).date()


def project_day_from_timestamp(value: str) -> date | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=PROJECT_TIMEZONE)
        return parsed.astimezone(PROJECT_TIMEZONE).date()
    except (TypeError, ValueError):
        return None


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from member3_dashboard.utils import database as db
from member3_dashboard.utils.omdb_matching import match_omdb_record, title_similarity


DEFAULT_LINKS = ROOT / "data" / "raw" / "ml-32m" / "links.csv"
DEFAULT_YOUTUBE = ROOT / "data" / "raw" / "ml-youtube.csv"
MOVIES_FILE = ROOT / "output" / "clean" / "movies_clean.csv"
OUTPUT_DIR = ROOT / "member2_recommendation" / "outputs"
MEDIA_DIR = ROOT / "member3_dashboard" / "media" / "metadata"
MANIFEST = MEDIA_DIR / "movie_media.csv"
CACHE = MEDIA_DIR / "omdb_media_cache.json"
PROGRESS = MEDIA_DIR / "acquisition_progress.json"
REPORT = MEDIA_DIR / "media_acquisition_report.json"
OMDB_URL = "https://www.omdbapi.com/"
YOUTUBE_OEMBED = "https://www.youtube.com/oembed"

MANIFEST_COLUMNS = [
    "movieId", "title", "release_year", "poster_path", "poster_url",
    "poster_source", "poster_status", "licensing_status", "trailer_url",
    "trailer_channel", "trailer_source", "trailer_status",
    "verification_status", "last_error", "overview", "imdbId",
    "oembed_status", "oembed_title", "oembed_author", "oembed_candidate_url",
    "poster_error", "trailer_error", "omdb_title", "omdb_year", "trailer_search_url",
    "fetched_at", "error_attempts", "low_confidence", "title_similarity", "match_reason",
]

# Previously reviewed official trailers from the existing CineSphere manifest.
# Keep them intact even when the optional MovieLens video mapping has no row.
PRESERVED_VERIFIED_TRAILERS = {
    171011: {
        "trailer_url": "https://www.youtube.com/watch?v=c8aFcHFu8QM",
        "trailer_channel": "BBC Earth",
        "trailer_source": "BBC Earth official channel; previously verified title and channel metadata",
        "trailer_status": "verified",
        "verification_status": "Official BBC Earth trailer page identified and opened previously; playback was not tested.",
    },
}


def read_env_key(path: Path) -> str:
    """Read the key without logging or returning it to console output."""
    if not path.is_file():
        return os.environ.get("OMDB_API_KEY", "").strip()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if stripped.startswith("OMDB_API_KEY="):
            value = stripped.partition("=")[2].strip().strip("\"'")
            return value or os.environ.get("OMDB_API_KEY", "").strip()
    return os.environ.get("OMDB_API_KEY", "").strip()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def imdb_id(value) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    digits = re.sub(r"\D", "", text)
    return f"tt{digits.zfill(7)}" if digits else ""


def youtube_id(value) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    match = re.search(r"(?:v=|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{6,})", text)
    return match.group(1) if match else text


def tier1_movie_ids() -> list[int]:
    """Return distinct IDs from the three files Member 2 feeds to the UI."""
    ordered: list[int] = []
    seen: set[int] = set()

    def add(values) -> None:
        for raw in values:
            value = pd.to_numeric(raw, errors="coerce")
            if pd.notna(value):
                movie_id = int(value)
                if movie_id not in seen:
                    seen.add(movie_id)
                    ordered.append(movie_id)

    popular = pd.read_csv(OUTPUT_DIR / "top_popular_movies.csv")
    als = pd.read_csv(OUTPUT_DIR / "pilot_user_recommendations.csv")
    genre = pd.read_csv(OUTPUT_DIR / "top_genre_movies.csv")
    for frame in (popular, als, genre):
        if "movieId" not in frame:
            raise ValueError("A Member 2 CSV is missing its movieId column.")
    add(popular["movieId"])
    add(als.sort_values(["userId", "rank", "movieId"], kind="stable")["movieId"])
    add(genre.sort_values(["genre", "rank", "movieId"], kind="stable")["movieId"])
    return ordered


def load_manifest() -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    if MANIFEST.is_file():
        with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    rows[int(row["movieId"])] = {k: (v or "") for k, v in row.items()}
                except (KeyError, TypeError, ValueError):
                    continue
    return rows


def write_manifest(rows: dict[int, dict[str, str]]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for movie_id in sorted(rows):
            row = {key: rows[movie_id].get(key, "") for key in MANIFEST_COLUMNS}
            row["movieId"] = movie_id
            writer.writerow(row)


def title_match(movie_title: str, video_title: str) -> bool:
    """Match a MovieLens title against trailer metadata after normalization."""
    stop_video = re.compile(r"\b(interview|critics? picks|episode|review|analysis|explained|commentary|conversation|talks|profile|featurette|behind the scenes|making of)\b", re.I)
    if stop_video.search(video_title) or not re.search(r"\b(trailer|teaser)\b", video_title, re.I):
        return False
    cleaned_video = re.sub(r"\b(?:official|trailer|teaser|hd|4k|original|theatrical)\b", " ", str(video_title), flags=re.I)
    cleaned_video = re.sub(r"\bdirector'?s\s+cut\b", " ", cleaned_video, flags=re.I)
    cleaned_video = re.sub(r"\b(?:19|20)\d{2}\b", " ", cleaned_video)
    cleaned_video = re.sub(r"\s+", " ", cleaned_video).strip(" -:|")
    aliases = [re.sub(r"\s*\((?:19|20)\d{2}\)\s*$", "", str(movie_title)).strip()]
    aliases.extend(re.findall(r"\(([^()]*)\)", aliases[0]))
    aliases.append(re.sub(r"\s*\([^()]*\)", " ", aliases[0]).strip())
    return any(title_similarity(alias, cleaned_video) >= 0.8 for alias in aliases if alias)


def youtube_search_url(title: str, year="") -> str:
    query = f"{title} {year} official trailer".strip()
    return "https://www.youtube.com/results?search_query=" + quote_plus(query)


def same_catalog_title(movie_title: str, source_title: str) -> bool:
    """Compare catalogue and OMDb titles with article movement and fuzzy edits."""
    return title_similarity(movie_title, source_title) >= 0.8


def reconcile_tier1(rows: dict[int, dict], cache: dict[str, dict], movies: pd.DataFrame,
                    youtube_map: dict[int, str], all_ids: list[int]) -> dict:
    movie_map = movies.set_index(movies.movieId.astype(int)).to_dict("index")
    poster_review_ids = []
    poster_title_mismatch = 0
    poster_year_mismatch = 0
    trailer_matches = 0
    trailer_mismatches = 0
    trailer_403 = 0
    trailer_401 = 0
    trailer_404 = 0
    trailer_link_only = 0
    trailer_search_links = 0
    trailer_no_candidate = 0
    raw_title_mismatches = 0
    recovered_title_mismatches = 0

    for movie_id in all_ids:
        record = rows.setdefault(movie_id, {"movieId": movie_id})
        if movie_id in PRESERVED_VERIFIED_TRAILERS:
            record.update(PRESERVED_VERIFIED_TRAILERS[movie_id])
        movie = movie_map[movie_id]
        title = str(movie.get("title", ""))
        year_value = movie.get("year")
        movie_year = int(year_value) if pd.notna(year_value) else None
        record["title"] = title
        record["release_year"] = movie_year or ""
        result = cache.get(str(movie_id), {})
        if result.get("poster_status") == "verified":
            omdb_title = str(result.get("omdb_title", ""))
            omdb_year_raw = str(result.get("omdb_year", ""))
            year_match = bool(movie_year and re.search(rf"\b{movie_year}\b", omdb_year_raw))
            title_match_exact = bool(omdb_title) and same_catalog_title(title, omdb_title)
            poster_title_mismatch += int(not title_match_exact)
            poster_year_mismatch += int(not year_match)
            # Poster QA is conservative: the returned OMDb title/year must identify
            # this catalogue entry; a title substring alone is insufficient.
            if not (title_match_exact and year_match):
                record["poster_status"] = "mismatch_review"
                record["poster_error"] = "OMDb title/year needs manual comparison with MovieLens."
                poster_review_ids.append(movie_id)
            else:
                record["poster_status"] = "verified"
            record["poster_url"] = result.get("poster_url", record.get("poster_url", ""))
            record["poster_source"] = "OMDb (IMDb ID matched)"
            record["licensing_status"] = "OMDb poster URL hotlinked for the approved non-commercial academic project; no local copy"
            record["omdb_title"] = omdb_title
            record["omdb_year"] = omdb_year_raw
            record["poster_path"] = ""

        raw_status = result.get("oembed_status", "")
        video_id = youtube_map.get(movie_id, "")
        candidate_url = result.get("oembed_candidate_url", "")
        if not candidate_url and video_id:
            candidate_url = f"https://www.youtube.com/watch?v={quote(video_id, safe='')}"
        if raw_status == "available_title_matched":
            # Revalidate old title matches with the stricter normalizer too.
            otitle = str(result.get("oembed_title", ""))
            if title_match(title, otitle):
                record.update({
                    "oembed_status": "available_title_matched",
                    "oembed_title": otitle,
                    "oembed_author": result.get("oembed_author", result.get("trailer_channel", "")),
                    "oembed_candidate_url": candidate_url,
                })
                if record.get("trailer_status") != "verified":
                    record.update({
                        "trailer_url": candidate_url,
                        "trailer_status": "candidate_unverified",
                        "trailer_source": "MovieLens YouTube mapping; checked with YouTube oEmbed",
                        "trailer_channel": result.get("oembed_author", result.get("trailer_channel", "")),
                    })
                trailer_matches += 1
            else:
                record["oembed_status"] = "title_mismatch"
                record["oembed_candidate_url"] = candidate_url
                if record.get("trailer_status") != "verified":
                    record.update({"trailer_url": "", "trailer_status": "unavailable_confirmed"})
                record["trailer_error"] = "oEmbed title failed normalized movie-title check; candidate suppressed."
                trailer_mismatches += 1
        elif raw_status == "title_mismatch":
            raw_title_mismatches += 1
            otitle = str(result.get("oembed_title", ""))
            record["oembed_title"] = otitle
            record["oembed_author"] = result.get("oembed_author", result.get("trailer_channel", ""))
            record["oembed_candidate_url"] = candidate_url
            if title_match(title, otitle):
                recovered_title_mismatches += 1
                record.update({
                    "oembed_status": "normalized_title_match",
                    "trailer_url": candidate_url,
                    "trailer_status": "candidate_unverified",
                    "trailer_source": "MovieLens YouTube mapping; checked with YouTube oEmbed",
                    "trailer_channel": result.get("oembed_author", result.get("trailer_channel", "")),
                    "trailer_error": "",
                })
                trailer_matches += 1
            else:
                record["oembed_status"] = "title_mismatch"
                record["oembed_candidate_url"] = candidate_url
                if record.get("trailer_status") != "verified":
                    record.update({"trailer_url": "", "trailer_status": "unavailable_confirmed"})
                record["trailer_error"] = "oEmbed title failed normalized movie-title check; candidate suppressed."
                trailer_mismatches += 1
        elif raw_status in {"http_401", "http_403"}:
            if raw_status == "http_401": trailer_401 += 1
            else: trailer_403 += 1
            record["oembed_status"] = raw_status
            record["oembed_candidate_url"] = candidate_url
            if record.get("trailer_status") != "verified":
                record.update({
                    "trailer_url": candidate_url,
                    "trailer_status": "link_only" if candidate_url else "unavailable_confirmed",
                    "trailer_source": "MovieLens YouTube mapping; oEmbed access restricted" if candidate_url else record.get("trailer_source", ""),
                    "trailer_error": "YouTube oEmbed returned " + raw_status[-3:] + "; link-only, not embedded.",
                })
            if candidate_url and record.get("trailer_status") == "link_only": trailer_link_only += 1
        elif raw_status == "http_404":
            trailer_404 += 1
            record["oembed_status"] = "http_404"
            record["oembed_candidate_url"] = candidate_url
            if record.get("trailer_status") != "verified":
                record.update({"trailer_url": "", "trailer_status": "unavailable_confirmed", "trailer_error": "YouTube oEmbed returned 404; candidate removed."})
        elif not video_id:
            search_url = youtube_search_url(title, movie_year or "")
            record["oembed_status"] = "no_mapping_candidate"
            record["oembed_candidate_url"] = ""
            record["trailer_search_url"] = search_url
            if record.get("trailer_status") != "verified":
                record.update({
                    "trailer_url": search_url,
                    "trailer_status": "link_only",
                    "trailer_source": "YouTube search link; no MovieLens trailer mapping",
                    "trailer_channel": "",
                })
            trailer_search_links += 1
        elif record.get("trailer_status") == "verified":
            pass
        else:
            record.update({"oembed_status": raw_status or "not_checked", "oembed_candidate_url": candidate_url})
            trailer_no_candidate += 1
    return {
        "poster_urls_returned": sum(cache.get(str(mid), {}).get("poster_status") == "verified" for mid in all_ids),
        "poster_mismatch_review": len(poster_review_ids),
        "poster_mismatch_review_movie_ids": poster_review_ids,
        "poster_title_mismatch": poster_title_mismatch,
        "poster_year_mismatch": poster_year_mismatch,
        "poster_usable_urls": max(0, sum(cache.get(str(mid), {}).get("poster_status") == "verified" for mid in all_ids) - len(poster_review_ids)),
        "normalized_oembed_title_matches": trailer_matches,
        "raw_title_mismatches": raw_title_mismatches,
        "recovered_title_mismatches": recovered_title_mismatches,
        "raw_mismatches_remaining": raw_title_mismatches - recovered_title_mismatches,
        "oembed_mismatches_suppressed": trailer_mismatches,
        "oembed_http_401": trailer_401,
        "oembed_http_403": trailer_403,
        "oembed_http_404_dropped": trailer_404,
        "oembed_restricted_link_only": trailer_link_only,
        "youtube_search_link_only": trailer_search_links,
        "trailer_candidates_without_final_status": trailer_no_candidate,
    }


def oembed_candidate(session: requests.Session, youtube_id: str, movie_title: str) -> dict:
    video_url = f"https://www.youtube.com/watch?v={quote(youtube_id, safe='')}"
    try:
        response = session.get(
            YOUTUBE_OEMBED,
            params={"url": video_url, "format": "json"},
            timeout=(8, 18),
        )
    except requests.RequestException as exc:
        return {"oembed_status": "request_error", "trailer_error": type(exc).__name__, "oembed_candidate_url": video_url}
    if response.status_code != 200:
        return {"oembed_status": f"http_{response.status_code}", "trailer_error": f"YouTube oEmbed HTTP {response.status_code}", "oembed_candidate_url": video_url}
    try:
        payload = response.json()
    except ValueError:
        return {"oembed_status": "invalid_json", "trailer_error": "YouTube oEmbed returned invalid JSON", "oembed_candidate_url": video_url}
    video_title = str(payload.get("title", "")).strip()
    author = str(payload.get("author_name", "")).strip()
    if not video_title or not author:
        return {"oembed_status": "missing_metadata", "trailer_error": "YouTube oEmbed omitted title or channel", "oembed_candidate_url": video_url}
    if not title_match(movie_title, video_title):
        return {
            "oembed_status": "title_mismatch",
            "oembed_title": video_title,
            "trailer_channel": author,
            "oembed_author": author,
            "oembed_candidate_url": video_url,
            "trailer_error": "YouTube oEmbed title did not match the MovieLens title",
        }
    return {
        "oembed_status": "available_title_matched",
        "oembed_title": video_title,
        "oembed_author": author,
        "oembed_candidate_url": video_url,
        "trailer_channel": author,
        "trailer_url": video_url,
        "trailer_source": "MovieLens YouTube mapping; checked with YouTube oEmbed",
        # oEmbed confirms availability and metadata, not official channel status.
        "trailer_status": "candidate_unverified",
        "trailer_error": "",
    }


def poster_fields(session: requests.Session, *, api_key: str, linked_imdb: str, year, timeout=(8, 18)) -> dict:
    if not linked_imdb:
        return {"poster_status": "no_imdb_id", "poster_error": "No IMDb ID in links.csv"}
    try:
        response = session.get(
            OMDB_URL,
            params={"i": linked_imdb, "apikey": api_key, "y": str(int(year)) if pd.notna(year) else ""},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return {"poster_status": "failed", "poster_error": type(exc).__name__, "request_failed": True}
    if response.status_code == 429:
        return {"poster_status": "daily_limit", "poster_error": "OMDb HTTP 429", "daily_limit": True}
    try:
        payload = response.json()
    except ValueError:
        return {"poster_status": "failed", "poster_error": f"OMDb HTTP {response.status_code}: invalid JSON", "request_failed": True}
    error = str(payload.get("Error", ""))
    if "request limit" in error.lower() or "daily limit" in error.lower():
        return {"poster_status": "daily_limit", "poster_error": "OMDb daily request limit reached", "daily_limit": True}
    if error.lower() == "invalid api key!":
        return {"poster_status": "invalid_key", "poster_error": "OMDb rejected the configured API key", "invalid_key": True}
    if response.status_code != 200 or payload.get("Response") != "True":
        return {"poster_status": "not_found", "poster_error": error or f"OMDb HTTP {response.status_code}"}
    response_id = imdb_id(payload.get("imdbID"))
    if response_id != linked_imdb:
        return {"poster_status": "failed", "poster_error": "OMDb IMDb ID did not match links.csv", "request_failed": True}
    poster_url = str(payload.get("Poster", "")).strip()
    if not poster_url or poster_url.upper() == "N/A" or not poster_url.startswith("https://"):
        return {
            "poster_status": "not_available",
            "poster_error": "OMDb returned no HTTPS poster URL",
            "overview": str(payload.get("Plot", "") or ""),
        }
    return {
        "poster_status": "verified",
        "poster_url": poster_url,
        "poster_source": "OMDb (IMDb ID matched)",
        "licensing_status": "OMDb poster URL hotlinked for the approved non-commercial academic project; no local copy",
        "poster_error": "",
        "overview": str(payload.get("Plot", "") or ""),
        "omdb_title": str(payload.get("Title", "")),
        "omdb_year": str(payload.get("Year", "")),
    }


def update_report(tier_ids: list[int], rows: dict[int, dict], stats: dict, *, status: str, next_id=None) -> None:
    existing = read_json(REPORT, {})
    tier_rows = [rows.get(mid, {}) for mid in tier_ids]
    existing.update({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tier1_target_movies": len(tier_ids),
        "tier1_processed_movies": stats["processed"],
        "tier1_remaining_movies": max(0, len(tier_ids) - stats["processed"]),
        "tier1_poster_urls": sum(r.get("poster_status") == "verified" and str(r.get("poster_source", "")).startswith("OMDb") for r in tier_rows),
        "tier1_poster_mismatch_review": sum(r.get("poster_status") == "mismatch_review" for r in tier_rows),
        "tier1_movies_without_omdb_poster": sum(r.get("poster_status") in {"not_available", "not_found"} for r in tier_rows),
        "tier1_movies_with_matched_oembed_candidate": sum(r.get("oembed_status") == "available_title_matched" for r in tier_rows),
        "tier1_movies_with_trailer_title_mismatch": sum(r.get("oembed_status") == "title_mismatch" for r in tier_rows),
        "tier1_oembed_http_failures": sum(str(r.get("oembed_status", "")).startswith("http_") for r in tier_rows),
        "tier1_youtube_mapping_candidates": sum(bool(r.get("oembed_candidate_url")) for r in tier_rows),
        "tier1_failed_requests": stats["failed"],
        "tier1_status": status,
        "tier1_resume_movieId": next_id,
        "tier1_poster_files_downloaded": 0,
        "tier1_browser_automation_used": False,
        "tier1_source": "OMDb API hotlinked poster URLs and MovieLens ml-youtube.csv validated with YouTube oEmbed",
        "tier1_note": "YouTube oEmbed validates availability/title/channel metadata, not that a channel is the official distributor. Such links remain candidate_unverified until official status is separately established.",
    })
    progress = read_json(PROGRESS, {})
    cache = read_json(CACHE, {}).get("results", {})
    catalog_ids = set(pd.to_numeric(pd.read_csv(MOVIES_FILE, usecols=["movieId"])["movieId"], errors="coerce").dropna().astype(int))
    tier1_set = set(tier_ids)
    tier2_ids = catalog_ids - tier1_set
    tier2_attempted = {int(mid) for mid in cache if str(mid).isdigit()} & tier2_ids
    manifest = load_manifest()
    tier2_poster_urls = sum(manifest.get(mid, {}).get("poster_status") == "verified" and bool(manifest.get(mid, {}).get("poster_url")) for mid in tier2_ids)
    tier2_mismatch_review = sum(manifest.get(mid, {}).get("poster_status") == "mismatch_review" for mid in tier2_ids)
    existing.update({
        "tier2_attempted_movies": len(tier2_attempted),
        "tier2_poster_urls": tier2_poster_urls,
        "tier2_poster_mismatch_review": tier2_mismatch_review,
        "tier2_poster_urls_usable": tier2_poster_urls,
    })
    existing["tier_coverage"] = {
        "tier1": {
            "target": len(tier_ids),
            "processed": stats["processed"],
            "remaining": max(0, len(tier_ids) - stats["processed"]),
            "poster_urls_returned": existing["tier1_poster_urls"],
            "poster_urls_usable": max(0, existing["tier1_poster_urls"] - existing["tier1_poster_mismatch_review"]),
            "poster_mismatch_review": existing["tier1_poster_mismatch_review"],
            "youtube_candidates_checked": progress.get("tier1_oembed_checked_count", 0),
            "status": status,
            "resume_movieId": next_id,
        },
        "tier2": {
            "target": len(tier2_ids),
            "processed": len(tier2_attempted),
            "remaining": max(0, len(tier2_ids) - len(tier2_attempted)),
            "attempted": len(tier2_attempted),
            "poster_urls": tier2_poster_urls,
            "poster_mismatch_review": tier2_mismatch_review,
            "poster_urls_usable": tier2_poster_urls,
            "status": progress.get("tier2_status", "not_started"),
            "next_offset": progress.get("tier2_offset", 0),
        },
    }
    atomic_json(REPORT, existing)


def update_tier2_report(target_count: int, offset: int, batch_size: int, status: str) -> None:
    report = read_json(REPORT, {})
    progress = read_json(PROGRESS, {})
    cache = read_json(CACHE, {}).get("results", {})
    ids = tier1_movie_ids()
    catalog_ids = set(pd.to_numeric(pd.read_csv(MOVIES_FILE, usecols=["movieId"])["movieId"], errors="coerce").dropna().astype(int))
    tier2_ids = catalog_ids - set(ids)
    attempted_ids = {int(mid) for mid in cache if str(mid).isdigit()} & tier2_ids
    manifest = load_manifest()
    poster_urls = sum(
        manifest.get(mid, {}).get("poster_status") == "verified"
        and bool(manifest.get(mid, {}).get("poster_url"))
        for mid in tier2_ids
    )
    mismatch_review = sum(manifest.get(mid, {}).get("poster_status") == "mismatch_review" for mid in tier2_ids)
    report.update({
        "tier2_target_movies": target_count,
        "tier2_processed_movies": min(target_count, offset),
        "tier2_attempted_movies": len(attempted_ids),
        "tier2_remaining_movies": max(0, target_count - offset),
        "tier2_poster_urls": poster_urls,
        "tier2_poster_mismatch_review": mismatch_review,
        "tier2_poster_urls_usable": poster_urls,
        "tier2_status": status,
        "tier2_next_offset": offset,
        "tier2_last_batch_size": batch_size,
    })
    report["tier_coverage"] = {
        "tier1": {
            "target": len(ids),
            "processed": int(progress.get("tier1_processed", 0)),
            "remaining": max(0, len(ids) - int(progress.get("tier1_processed", 0))),
            "poster_urls_returned": report.get("tier1_poster_urls", 0),
            "poster_urls_usable": max(0, report.get("tier1_poster_urls", 0) - report.get("tier1_poster_mismatch_review", 0)),
            "poster_mismatch_review": report.get("tier1_poster_mismatch_review", 0),
            "youtube_candidates_checked": progress.get("tier1_oembed_checked_count", 0),
            "status": progress.get("tier1_status", "complete"),
            "resume_movieId": progress.get("tier1_resume_movieId"),
        },
        "tier2": {
            "target": len(tier2_ids),
            "processed": min(target_count, offset),
            "attempted": len(attempted_ids),
            "remaining": max(0, len(tier2_ids) - offset),
            "poster_urls": poster_urls,
            "poster_mismatch_review": mismatch_review,
            "poster_urls_usable": poster_urls,
            "status": status,
            "next_offset": offset,
        },
    }
    atomic_json(REPORT, report)


def run(args) -> int:
    api_key = read_env_key(args.env_file)
    if not api_key and not args.verify_oembed_only and not args.reconcile_tier1:
        print("OMDB_KEY_EMPTY")
        return 2
    for path in (args.links, args.youtube, MOVIES_FILE):
        if not path.is_file():
            print(f"INPUT_MISSING: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.name}")
            return 2

    all_ids = tier1_movie_ids()
    if len(all_ids) != 409:
        print(f"TIER1_ID_COUNT_MISMATCH: expected approximately 409, found {len(all_ids)}")
        return 2
    movies = pd.read_csv(MOVIES_FILE)
    links = pd.read_csv(args.links, dtype={"imdbId": "string"})
    youtube = pd.read_csv(args.youtube, dtype={"youtubeId": "string"})
    required_links = {"movieId", "imdbId"}
    if not required_links.issubset(links.columns):
        print("INPUT_INVALID: links.csv needs movieId and imdbId columns")
        return 2
    if not {"movieId", "youtubeId"}.issubset(youtube.columns):
        print("INPUT_INVALID: ml-youtube.csv needs movieId and youtubeId columns")
        return 2

    for frame in (movies, links, youtube):
        frame["movieId"] = pd.to_numeric(frame["movieId"], errors="coerce")
    movies = movies.dropna(subset=["movieId"]).drop_duplicates("movieId").copy()
    links = links.dropna(subset=["movieId"]).drop_duplicates("movieId").copy()
    youtube = youtube.dropna(subset=["movieId", "youtubeId"]).copy()
    movie_map = movies.set_index(movies.movieId.astype(int)).to_dict("index")
    link_map = links.set_index(links.movieId.astype(int)).to_dict("index")
    youtube_map: dict[int, str] = {}
    for row in youtube.itertuples(index=False):
        mid = int(row.movieId)
        vid = youtube_id(row.youtubeId)
        if mid not in youtube_map and vid and vid.lower() != "nan":
            youtube_map[mid] = vid

    rows = load_manifest()
    cache_data = read_json(CACHE, {})
    cache: dict[str, dict] = cache_data.get("results", {}) if isinstance(cache_data, dict) else {}
    progress = read_json(PROGRESS, {})
    if args.verify_oembed_only:
        session = requests.Session()
        checked = set(map(int, progress.get("tier1_oembed_checked_movie_ids", [])))
        candidates = [mid for mid in all_ids[args.offset or 0:(args.offset or 0) + args.limit] if mid in youtube_map]
        for mid in candidates:
            row = rows.get(mid, {})
            if row.get("oembed_status") not in (None, "", "no_candidate", "request_error"):
                continue
            result = oembed_candidate(session, youtube_map[mid], str(movie_map.get(mid, {}).get("title", row.get("title", ""))))
            row.update({k: v for k, v in result.items() if k not in {"trailer_url", "trailer_source", "trailer_status", "trailer_channel"} and v is not None})
            if result.get("oembed_status") == "available_title_matched":
                if row.get("trailer_status") != "verified":
                    row.update({k: result[k] for k in ("trailer_url", "trailer_source", "trailer_status", "trailer_channel") if k in result})
                    row["verification_status"] = (row.get("verification_status", "") + " YouTube oEmbed responded; title matched; channel official status and playback not verified.").strip()
            elif result.get("trailer_error"):
                row["trailer_error"] = result["trailer_error"]
            rows[mid] = row
            checked.add(mid)
            print(f"oEmbed-only: movieId={mid} result={result.get('oembed_status')}", flush=True)
        progress["tier1_oembed_checked_movie_ids"] = sorted(checked)
        progress["tier1_oembed_checked_count"] = len(checked)
        progress["last_updated"] = datetime.now(timezone.utc).isoformat()
        atomic_json(PROGRESS, progress)
        write_manifest(rows)
        completed_count = int(progress.get("tier1_processed", 0))
        update_report(all_ids, rows, {"processed": completed_count, "failed": 0}, status=progress.get("tier1_status", "running"), next_id=progress.get("tier1_resume_movieId"))
        return 0

    if args.reconcile_tier1:
        stats = reconcile_tier1(rows, cache, movies, youtube_map, all_ids)
        write_manifest(rows)
        progress.update({
            "tier1_target_movies": len(all_ids),
            "tier1_processed": len(all_ids),
            "tier1_remaining_movie_ids": [],
            "tier1_resume_movieId": None,
            "tier1_status": "complete",
            "tier1_youtube_mapping_candidates": len(youtube_map.keys() & set(all_ids)),
            "tier1_reconciled_at": datetime.now(timezone.utc).isoformat(),
            "tier2_status": progress.get("tier2_status", "not_started"),
            "tier2_offset": progress.get("tier2_offset", 0),
        })
        atomic_json(PROGRESS, progress)
        update_report(all_ids, rows, {"processed": len(all_ids), "failed": 0}, status="complete", next_id=None)
        report = read_json(REPORT, {})
        report.update({
            "tier1_poster_mismatch_review": stats["poster_mismatch_review"],
            "tier1_poster_mismatch_review_movie_ids": stats["poster_mismatch_review_movie_ids"],
            "tier1_poster_title_mismatch": stats["poster_title_mismatch"],
            "tier1_poster_year_mismatch": stats["poster_year_mismatch"],
            "tier1_poster_usable_urls_after_qa": stats["poster_usable_urls"],
            "tier1_normalized_oembed_title_matches": stats["normalized_oembed_title_matches"],
            "tier1_raw_oembed_title_mismatches": stats["raw_title_mismatches"],
            "tier1_title_mismatches_recovered_by_normalization": stats["recovered_title_mismatches"],
            "tier1_title_mismatches_still_suppressed": stats["raw_mismatches_remaining"],
            "tier1_oembed_mismatches_suppressed": stats["oembed_mismatches_suppressed"],
            "tier1_oembed_http_401": stats["oembed_http_401"],
            "tier1_oembed_http_403": stats["oembed_http_403"],
            "tier1_oembed_http_404_dropped": stats["oembed_http_404_dropped"],
            "tier1_oembed_restricted_link_only": stats["oembed_restricted_link_only"],
            "tier1_youtube_search_link_only": stats["youtube_search_link_only"],
            "tier1_trailer_candidates_without_final_status": stats["trailer_candidates_without_final_status"],
        })
        report["tier_coverage"]["tier1"].update({
            "poster_urls_returned": stats["poster_urls_returned"],
            "poster_urls_usable": stats["poster_usable_urls"],
            "poster_mismatch_review": stats["poster_mismatch_review"],
        })
        atomic_json(REPORT, report)
        print(json.dumps({k: v for k, v in stats.items() if k != "poster_mismatch_review_movie_ids"}, indent=2))
        return 0

    completed = set(map(int, progress.get("tier1_completed_movie_ids", [])))
    target = all_ids
    if args.tier == "2":
        full_ids = [int(x) for x in movies.movieId]
        tier1_set = set(all_ids)
        target = [mid for mid in full_ids if mid not in tier1_set]
    offset = args.offset
    if args.tier == "2" and offset is None:
        offset = int(progress.get("tier2_offset", 0))
    offset = offset or 0
    batch = target[offset:offset + args.limit]
    if args.tier == "1":
        batch = [mid for mid in batch if mid not in completed]
    stats = {"processed": len(completed), "failed": 0}
    session = requests.Session()
    session.headers.update({"User-Agent": "CineSphere academic media reference checker"})
    stop_status = "complete"
    resume_id = None

    for position, mid in enumerate(batch):
        movie = movie_map.get(mid)
        if not movie:
            result = {"movieId": mid, "poster_status": "failed", "poster_error": "MovieLens ID absent from cleaned catalogue", "request_failed": True}
        else:
            title = str(movie.get("title", ""))
            year = movie.get("year", None)
            linked = link_map.get(mid, {})
            imdb = imdb_id(linked.get("imdbId", ""))
            result = {"movieId": mid, "title": title, "release_year": int(year) if pd.notna(year) else "", "imdbId": imdb}
            poster = poster_fields(session, api_key=api_key, linked_imdb=imdb, year=year)
            if poster.pop("daily_limit", False):
                result.update(poster)
                result["daily_limit"] = True
            elif poster.pop("invalid_key", False):
                result.update(poster)
                result["invalid_key"] = True
            else:
                poster.pop("request_failed", None)
                result.update(poster)
                if poster.get("poster_status") == "failed":
                    stats["failed"] += 1
                prior = rows.get(mid, {})
                # Preserve an earlier official/human-verified trailer.
                video_id = youtube_map.get(mid)
                if video_id:
                    trailer = oembed_candidate(session, video_id, title)
                    result.update(trailer)
                    if trailer.get("trailer_error") and trailer.get("oembed_status", "").startswith("request"):
                        stats["failed"] += 1
                else:
                    result["oembed_status"] = prior.get("oembed_status", "no_candidate")
                # Keep prior source-researched fields unless a new source returned a result.
                merged = dict(prior)
                merged.update({k: v for k, v in result.items() if v is not None and v != ""})
                merged["movieId"] = mid
                merged["title"] = title
                merged["release_year"] = result["release_year"]
                if result.get("poster_status") == "verified":
                    merged["poster_path"] = ""
                    merged["poster_url"] = result["poster_url"]
                    merged["poster_source"] = result["poster_source"]
                    merged["verification_status"] = f"OMDb record matched IMDb ID {imdb}; poster URL is hotlinked, not downloaded."
                if result.get("trailer_status") == "candidate_unverified":
                    if prior.get("trailer_status") != "verified":
                        merged["trailer_url"] = result["trailer_url"]
                        merged["trailer_source"] = result["trailer_source"]
                        merged["trailer_channel"] = result.get("trailer_channel", "")
                        merged["trailer_status"] = "candidate_unverified"
                    merged["oembed_status"] = result["oembed_status"]
                    merged["oembed_title"] = result.get("oembed_title", "")
                    merged["oembed_author"] = result.get("oembed_author", "")
                    merged["oembed_candidate_url"] = result.get("oembed_candidate_url", "")
                    if prior.get("trailer_status") != "verified":
                        merged["verification_status"] = (merged.get("verification_status", "") + " YouTube oEmbed responded; title matched; channel official status and playback not verified.").strip()
                if result.get("oembed_status") == "title_mismatch":
                    merged["oembed_status"] = "title_mismatch"
                    merged["trailer_error"] = result.get("trailer_error", "")
                if result.get("poster_error"):
                    merged["poster_error"] = result["poster_error"]
                if result.get("trailer_error"):
                    merged["last_error"] = result["trailer_error"]
                rows[mid] = merged
        cache[str(mid)] = result
        retryable_failure = result.get("poster_status") == "failed" or result.get("oembed_status") == "request_error"
        if result.get("daily_limit"):
            stop_status = "stopped_daily_limit"
            resume_id = mid
        elif result.get("invalid_key"):
            stop_status = "stopped_invalid_key"
            resume_id = mid
        else:
            if args.tier == "2" and retryable_failure:
                stop_status = "paused_resumable"
                resume_id = mid
            if args.tier == "1" and not retryable_failure:
                completed.add(mid)
                stats["processed"] = len(completed)
        if result.get("daily_limit") or result.get("invalid_key"):
            movie = movie_map.get(mid, {})
            merged = dict(rows.get(mid, {}))
            merged.update({
                "movieId": mid,
                "title": str(movie.get("title", merged.get("title", ""))),
                "release_year": int(movie["year"]) if movie and pd.notna(movie.get("year")) else merged.get("release_year", ""),
                "imdbId": imdb_id(link_map.get(mid, {}).get("imdbId", "")),
                "poster_status": result.get("poster_status", "failed"),
                "poster_error": result.get("poster_error", ""),
                "last_error": result.get("poster_error", ""),
            })
            rows[mid] = merged
        if args.tier == "1":
            progress.update({
                "tier1_target_movies": len(all_ids),
                "tier1_completed_movie_ids": sorted(completed),
                "tier1_resume_movieId": mid if stop_status != "complete" else next((x for x in all_ids if x not in completed), None),
                "tier1_status": stop_status if stop_status != "complete" else "running",
                "tier1_processed": len(completed),
                "tier1_remaining_movie_ids": [x for x in all_ids if x not in completed],
                "last_processed_movieId": mid,
                "last_attempted_movieId": mid,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            })
        else:
            next_offset = offset + position + (0 if stop_status != "complete" or retryable_failure else 1)
            progress.update({
                "tier2_offset": next_offset,
                "tier2_status": stop_status if stop_status != "complete" else "running",
                "tier2_last_movieId": mid,
                "last_attempted_movieId": mid,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            })
        atomic_json(CACHE, {"results": cache, "updated_at": datetime.now(timezone.utc).isoformat()})
        atomic_json(PROGRESS, progress)
        write_manifest(rows)
        if args.tier == "1":
            update_report(all_ids, rows, stats, status=stop_status if stop_status != "complete" else "running", next_id=resume_id or progress.get("tier1_resume_movieId"))
        else:
            update_tier2_report(len(target), int(progress.get("tier2_offset", offset)), position + 1, str(progress.get("tier2_status", "running")))
        print(f"Tier {args.tier}: {position + 1}/{len(batch)} movieId={mid} poster={result.get('poster_status', 'unchanged')} trailer={result.get('trailer_status', result.get('oembed_status', 'none'))}", flush=True)
        if stop_status != "complete":
            break
        if args.delay:
            time.sleep(args.delay)

    if args.tier == "1" and stop_status == "complete":
        stats["processed"] = len(completed)
        stop_status = "complete" if len(completed) == len(all_ids) else "paused_resumable"
        resume_id = next((x for x in all_ids if x not in completed), None)
        progress.update({
            "tier1_target_movies": len(all_ids),
            "tier1_completed_movie_ids": sorted(completed),
            "tier1_processed": len(completed),
            "tier1_remaining_movie_ids": [x for x in all_ids if x not in completed],
            "tier1_resume_movieId": resume_id,
            "tier1_status": stop_status,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "tier2_status": "not_started",
        })
        atomic_json(PROGRESS, progress)
        update_report(all_ids, rows, stats, status=stop_status, next_id=resume_id)
    elif args.tier == "2":
        if stop_status == "complete":
            next_offset = offset + len(batch)
            tier2_status = "complete" if next_offset >= len(target) else "paused_resumable"
            progress.update({"tier2_status": tier2_status, "tier2_offset": next_offset, "tier2_batch_size": len(batch)})
        else:
            next_offset = int(progress.get("tier2_offset", offset))
            tier2_status = stop_status
        update_tier2_report(len(target), next_offset, len(batch), tier2_status)
        atomic_json(PROGRESS, progress)
    return 0 if stop_status == "complete" else (3 if stop_status == "stopped_daily_limit" else 2 if stop_status == "stopped_invalid_key" else 0)


def _canonical_media_status(record: dict) -> str:
    status = str(record.get("poster_status", "")).lower()
    if status in {"ok", "verified"} and record.get("poster_url") and str(record.get("poster_source", "")).lower().startswith("omdb"):
        return "ok"
    if status in {"poster_na", "not_available"}:
        return "poster_na"
    if status == "no_imdb_id":
        return status
    if status == "not_found":
        return status
    if status == "mismatch_review":
        return status
    if status in {"error", "failed", "daily_limit"}:
        return "error"
    if status == "" and not record:
        return ""
    return status or ""


def _reconcile_manifest_record(record: dict, movie: dict, linked_imdb: str) -> tuple[dict, bool]:
    before = _canonical_media_status(record)
    title = str(movie.get("title", ""))
    year = movie.get("year")
    record.update({"movieId": int(movie["movieId"]), "title": title, "imdbId": linked_imdb,
                   "release_year": int(year) if pd.notna(year) else ""})
    if before == "ok" or (record.get("poster_url") and record.get("omdb_title") and str(record.get("poster_source", "")).lower().startswith("omdb")):
        matched, low_confidence, similarity, reason = match_omdb_record(
            title, year, str(record.get("omdb_title", "")), str(record.get("omdb_year", ""))
        )
        record.update({"poster_status": "ok" if matched else "mismatch_review",
                       "low_confidence": str(bool(low_confidence)).lower(),
                       "title_similarity": f"{similarity:.4f}", "match_reason": reason,
                       "poster_error": "" if matched else reason})
    elif before:
        record["poster_status"] = before
    return record, before != _canonical_media_status(record)


def _coverage_report(movies: pd.DataFrame, links: pd.DataFrame, rows: dict[int, dict],
                     progress: dict, changed_rows: int, daily_limit: int) -> dict:
    link_map = links.set_index(links.movieId.astype(int))["imdbId"].to_dict()
    catalog = movies.copy()
    catalog["movieId"] = catalog.movieId.astype(int)
    catalog["rating_count"] = pd.to_numeric(catalog.rating_count, errors="coerce").fillna(0)
    catalog["has_imdb_id"] = catalog.movieId.map(lambda mid: bool(imdb_id(link_map.get(mid, ""))))

    def counts_for(ids) -> dict:
        subset_rows = [rows.get(int(mid), {}) for mid in ids]
        statuses = [_canonical_media_status(row) for row in subset_rows]
        count = len(subset_rows)
        imdb_count = sum(bool(imdb_id(link_map.get(int(mid), ""))) for mid in ids)
        recognized = {"ok", "not_found", "poster_na", "mismatch_review", "no_imdb_id", "error", "pending", ""}
        return {
            "catalog_movies": count,
            "with_imdb_id": imdb_count,
            "ok": statuses.count("ok"),
            "not_found": statuses.count("not_found"),
            "poster_na": statuses.count("poster_na"),
            "mismatch_review": statuses.count("mismatch_review"),
            "no_imdb_id": statuses.count("no_imdb_id"),
            "error": statuses.count("error"),
            "unprocessed": sum(status in {"", "pending"} for status in statuses),
            "other_legacy_status": sum(status not in recognized for status in statuses),
            "eligible_remaining": max(0, imdb_count - statuses.count("ok") - statuses.count("not_found") - statuses.count("poster_na") - statuses.count("mismatch_review")),
            "days_needed_at_daily_limit": None,
        }

    overall = counts_for(catalog.movieId.tolist())
    overall["days_needed_at_daily_limit"] = ceil(overall["eligible_remaining"] / daily_limit) if daily_limit else None
    error_types = {}
    trailer_statuses = [str(row.get("oembed_status", "")) for row in rows.values()]
    trailer_records = list(rows.values())
    trailer_coverage = {
        "oembed_http_401_link_only": trailer_statuses.count("http_401"),
        "oembed_http_403_link_only": trailer_statuses.count("http_403"),
        "oembed_http_404_dropped": trailer_statuses.count("http_404"),
        "no_mapping_candidates": trailer_statuses.count("no_mapping_candidate"),
        "youtube_search_fallbacks": sum(bool(row.get("trailer_search_url")) for row in trailer_records),
        "youtube_search_link_only": sum(
            str(row.get("trailer_status", "")) == "link_only"
            and str(row.get("trailer_source", "")).startswith("YouTube search link")
            for row in trailer_records
        ),
        "candidate_unverified": sum(str(row.get("trailer_status", "")) == "candidate_unverified" for row in trailer_records),
        "official_verified": sum(str(row.get("trailer_status", "")) == "verified" for row in trailer_records),
        "suppressed_title_mismatch_total": trailer_statuses.count("title_mismatch"),
        "suppressed_title_mismatches_with_trailer_url": sum(
            str(row.get("oembed_status", "")) == "title_mismatch" and bool(row.get("trailer_url"))
            for row in trailer_records
        ),
    }
    for row in rows.values():
        if _canonical_media_status(row) == "error":
            label = str(row.get("poster_error", "unspecified")) or "unspecified"
            error_types[label] = error_types.get(label, 0) + 1
    by_cutoff = {}
    for cutoff in (1000, 300, 100, 50):
        ids = catalog.loc[catalog.rating_count >= cutoff, "movieId"].tolist()
        data = counts_for(ids)
        data["cutoff_rating_count"] = cutoff
        data["full_catalog_days_at_daily_limit"] = ceil(data["with_imdb_id"] / daily_limit) if daily_limit else None
        data["eligible_remaining"] = max(0, data["with_imdb_id"] - data["ok"] - data["not_found"] - data["poster_na"] - data["mismatch_review"])
        data["days_needed_at_daily_limit"] = ceil(data["eligible_remaining"] / daily_limit) if daily_limit else None
        by_cutoff[str(cutoff)] = data
    today = project_today().isoformat()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "OMDb IMDb ID lookup; poster URLs hotlinked; no poster files downloaded",
        "daily_limit": daily_limit,
        "today": today,
        "today_reserved_requests": db.omdb_usage_count(today),
        "resume_movieId": progress.get("resume_movieId"),
        "resume_status": progress.get("status", "ready"),
        "rows_changed_by_current_matcher": changed_rows,
        "overall": overall,
        "error_types": error_types,
        "trailer_coverage": trailer_coverage,
        "coverage_by_rating_count": by_cutoff,
        "legacy_trailer_title_mismatch_recheck": progress.get("trailer_title_recheck", {}),
        "poster_files_downloaded": 0,
        "browser_automation_used": False,
        "per_movie_web_search_used": False,
    }


def run_priority(args) -> int:
    if not args.links.is_file() or not MOVIES_FILE.is_file():
        print("INPUT_MISSING: links.csv or output/clean/movies_clean.csv")
        return 2
    movies = pd.read_csv(MOVIES_FILE)
    links = pd.read_csv(args.links, dtype={"imdbId": "string"})
    youtube = pd.read_csv(args.youtube, dtype={"youtubeId": "string"}) if args.youtube.is_file() else pd.DataFrame()
    if not {"movieId", "imdbId"}.issubset(links.columns):
        print("INPUT_INVALID: links.csv needs movieId and imdbId columns")
        return 2
    if not youtube.empty and not {"movieId", "youtubeId"}.issubset(youtube.columns):
        print("INPUT_INVALID: ml-youtube.csv needs movieId and youtubeId columns")
        return 2
    movies["movieId"] = pd.to_numeric(movies.movieId, errors="coerce")
    movies["rating_count"] = pd.to_numeric(movies.get("rating_count", 0), errors="coerce").fillna(0)
    movies = movies.dropna(subset=["movieId"]).drop_duplicates("movieId").copy()
    movies["movieId"] = movies.movieId.astype(int)
    links["movieId"] = pd.to_numeric(links.movieId, errors="coerce")
    links = links.dropna(subset=["movieId"]).drop_duplicates("movieId").copy()
    link_map = links.set_index(links.movieId.astype(int))["imdbId"].to_dict()
    youtube_map: dict[int, str] = {}
    if not youtube.empty:
        youtube["movieId"] = pd.to_numeric(youtube.movieId, errors="coerce")
        youtube = youtube.dropna(subset=["movieId", "youtubeId"]).drop_duplicates("movieId")
        for item in youtube.itertuples(index=False):
            video_id = youtube_id(item.youtubeId)
            if re.fullmatch(r"[A-Za-z0-9_-]{6,}", video_id):
                youtube_map[int(item.movieId)] = video_id
    rows = load_manifest()
    cache_file = read_json(CACHE, {})
    cache = cache_file.get("results", {}) if isinstance(cache_file, dict) else {}
    original_poster_statuses = {mid: _canonical_media_status(row) for mid, row in rows.items()}
    with db.connect() as conn:
        sqlite_media = [dict(row) for row in conn.execute("SELECT * FROM movie_media_cache")]
    for cached in sqlite_media:
        mid = int(cached["movie_id"])
        if mid not in rows or not str(rows[mid].get("poster_status", "") or "").strip():
            rows[mid] = {"movieId": mid, "poster_status": cached["status"],
                         "poster_url": cached["poster_url"], "overview": cached["plot_text"],
                         "fetched_at": cached["fetched_at"], "poster_source": "OMDb (IMDb ID matched)"}
    progress = read_json(PROGRESS, {})
    movie_map = movies.set_index("movieId").to_dict("index")
    changed = 0
    trailer_recovered = 0
    trailer_suppressed = 0
    raw_trailer_mismatch_count = 0
    for mid_text, result in cache.items():
        if result.get("oembed_status") != "title_mismatch":
            continue
        try:
            mid = int(mid_text)
        except (TypeError, ValueError):
            continue
        if mid not in movie_map:
            continue
        raw_trailer_mismatch_count += 1
        record = rows.setdefault(mid, {"movieId": mid})
        record["oembed_title"] = result.get("oembed_title", "")
        record["oembed_candidate_url"] = result.get("oembed_candidate_url", "")
        author = result.get("oembed_author", "")
        record["oembed_author"] = author
        candidate_url = result.get("oembed_candidate_url", "")
        if title_match(str(movie_map[mid].get("title", "")), str(result.get("oembed_title", ""))):
            record.update({"oembed_status": "available_title_matched", "trailer_url": candidate_url,
                           "trailer_status": "candidate_unverified", "trailer_channel": author,
                           "trailer_source": "MovieLens YouTube mapping; normalized oEmbed title match; source unverified",
                           "trailer_error": ""})
            trailer_recovered += 1
        else:
            if record.get("trailer_status") != "verified":
                record.update({"trailer_url": "", "trailer_status": "unavailable_confirmed"})
            record["oembed_status"] = "title_mismatch"
            record["trailer_error"] = "YouTube oEmbed title did not match normalized MovieLens title."
            trailer_suppressed += 1
    progress["trailer_title_recheck"] = {
        "old_title_mismatch_candidates": raw_trailer_mismatch_count,
        "recovered_by_normalization": trailer_recovered,
        "still_suppressed": trailer_suppressed,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    for mid, movie in movie_map.items():
        movie["movieId"] = mid
        record = rows.setdefault(mid, {"movieId": mid})
        linked = imdb_id(link_map.get(mid, ""))
        cached = cache.get(str(mid), {})
        if cached:
            for key in ("poster_url", "poster_source", "omdb_title", "omdb_year", "overview", "poster_error"):
                if cached.get(key) not in (None, ""):
                    record[key] = cached[key]
            if cached.get("poster_status"):
                record["poster_status"] = cached["poster_status"]
            record.setdefault("fetched_at", cached.get("fetched_at", ""))
            if cached.get("error_attempts") is not None:
                record["error_attempts"] = cached["error_attempts"]
            elif cached.get("poster_status") == "error" and project_day_from_timestamp(cached.get("fetched_at", "")) == project_today():
                record["error_attempts"] = 3
        if not linked:
            record.update({"poster_status": "no_imdb_id", "poster_url": "", "imdbId": ""})
        elif record.get("poster_url") and record.get("omdb_title"):
            record, _ = _reconcile_manifest_record(record, movie, linked)
            previous_status = original_poster_statuses.get(mid, "")
            new_status = _canonical_media_status(record)
            changed += int(previous_status in {"ok", "mismatch_review", "not_found", "poster_na", "no_imdb_id", "error"}
                           and previous_status != new_status)
        elif record.get("poster_status"):
            record["poster_status"] = _canonical_media_status(record)
        record["movieId"] = mid
        record["title"] = str(movie.get("title", ""))
        record["release_year"] = int(movie["year"]) if pd.notna(movie.get("year")) else ""
        # Preserve existing reviewed/checked statuses. Seed only absent trailer
        # records from the downloaded MovieLens mapping as link-only candidates.
        if youtube_map.get(mid) and not record.get("trailer_status") and not record.get("trailer_url"):
            record.update({
                "trailer_url": f"https://www.youtube.com/watch?v={quote(youtube_map[mid], safe='')}",
                "trailer_status": "link_only",
                "trailer_source": "MovieLens YouTube mapping; not rechecked",
                "trailer_error": "Candidate link only; availability and title have not been checked.",
            })
    write_manifest(rows)

    member2_ids = [mid for mid in tier1_movie_ids() if mid in movie_map]
    preferred = set(member2_ids)
    rest = movies.loc[~movies.movieId.isin(preferred)].sort_values(
        ["rating_count", "movieId"], ascending=[False, True], kind="stable").movieId.tolist()
    queue = [*member2_ids, *rest]
    counts = {cutoff: int((movies.rating_count >= cutoff).sum()) for cutoff in (1000, 300, 100, 50)}
    imdb_count = sum(bool(imdb_id(link_map.get(mid, ""))) for mid in movie_map)
    print("Priority queue: Member 2 recommendation IDs first, then catalogue by rating_count descending.")
    for cutoff, count in counts.items():
        linked_count = sum(bool(imdb_id(link_map.get(int(mid), ""))) for mid in movies.loc[movies.rating_count >= cutoff, "movieId"])
        print(f"rating_count>={cutoff}: movies={count}, with_imdbId={linked_count}, days_at_{args.daily_limit}/day={ceil(linked_count / args.daily_limit) if args.daily_limit else 'n/a'}")
    print(f"catalogue_movies={len(movie_map)}, with_imdbId={imdb_count}")

    if args.seed_youtube_links_only:
        progress.update({"last_run_date": project_today().isoformat(),
                         "youtube_mapping_links_seeded": sum(row.get("trailer_source") == "MovieLens YouTube mapping; not rechecked" for row in rows.values()),
                         "youtube_mapping_links_seeded_at": datetime.now(timezone.utc).isoformat()})
        report = _coverage_report(movies, links, rows, progress, changed, args.daily_limit)
        atomic_json(PROGRESS, progress)
        atomic_json(REPORT, report)
        print(f"YouTube mapping links available: {progress['youtube_mapping_links_seeded']}")
        print("Links are marked unverified and will not be embedded in the dashboard.")
        return 0

    api_key = read_env_key(args.env_file)
    if not api_key:
        progress.update({"status": "key_empty", "resume_movieId": None, "match_rechecks_changed": changed})
        atomic_json(PROGRESS, progress)
        report = _coverage_report(movies, links, rows, progress, changed, args.daily_limit)
        atomic_json(REPORT, report)
        print("OMDB_KEY_EMPTY")
        return 2

    today = project_today().isoformat()
    initial_usage = db.omdb_usage_count(today)
    requested_today = 0
    saved = 0
    stop = "complete"
    resume = None
    prior_proxy_failures = sum(
        result.get("poster_status") == "error"
        and result.get("poster_error") == "ProxyError"
        and project_day_from_timestamp(result.get("fetched_at", "")) == project_today()
        for result in cache.values()
    )
    probe_cached_proxy_failures = prior_proxy_failures > 0
    session = requests.Session()
    session.headers.update({"User-Agent": "CineSphere academic media reference checker"})

    for position, mid in enumerate(queue):
        record = rows[mid]
        linked = imdb_id(link_map.get(mid, ""))
        status = _canonical_media_status(record)
        if not linked or status in {"ok", "no_imdb_id", "poster_na", "mismatch_review", "pending"}:
            continue
        fetched_day = project_day_from_timestamp(record.get("fetched_at", ""))
        fetched_at = fetched_day.isoformat() if fetched_day else ""
        if status == "not_found" and fetched_day:
            if fetched_day >= project_today() - timedelta(days=90):
                continue
        if status == "error" and fetched_at == today and int(record.get("error_attempts", 0) or 0) >= 3:
            continue
        if db.omdb_usage_count(today) >= args.daily_limit:
            stop, resume = "daily_limit_reached", mid
            break
        result = {}
        attempt_number = int(record.get("error_attempts", 0) or 0) if fetched_at == today else 0
        # A saved proxy failure must not permanently block retries for the day.
        # Probe once before resuming the queue; stop again if connectivity is
        # still down, otherwise proceed with the normal per-title retry budget.
        max_attempts = 1 if probe_cached_proxy_failures else max(0, 3 - attempt_number)
        for _ in range(max_attempts):
            if not db.reserve_omdb_request(args.daily_limit):
                stop, resume = "daily_limit_reached", mid
                break
            requested_today += 1
            result = poster_fields(session, api_key=api_key, linked_imdb=linked,
                                  year=movie_map[mid].get("year"), timeout=(5, 12))
            if result.get("daily_limit"):
                stop, resume = "daily_limit_reached", mid
                break
            if result.get("invalid_key"):
                stop, resume = "invalid_key", mid
                break
            if not result.get("request_failed"):
                probe_cached_proxy_failures = False
                prior_proxy_failures = 0
                break
            attempt_number += 1
            if probe_cached_proxy_failures:
                stop, resume = "network_unavailable", mid
                break
            if attempt_number >= 3:
                break
        if stop in {"daily_limit_reached", "invalid_key"}:
            break

        fetched = datetime.now(timezone.utc).isoformat()
        raw_status = result.get("poster_status", "error")
        if raw_status == "verified":
            accepted, low_confidence, similarity, reason = match_omdb_record(
                movie_map[mid].get("title", ""), movie_map[mid].get("year"),
                result.get("omdb_title", ""), result.get("omdb_year", "")
            )
            final_status = "ok" if accepted else "mismatch_review"
            result.update({"low_confidence": str(bool(low_confidence)).lower(),
                           "title_similarity": f"{similarity:.4f}", "match_reason": reason})
        elif raw_status == "not_available":
            final_status = "poster_na"
        elif raw_status == "not_found":
            final_status = "not_found"
        else:
            final_status = "error"
        record.update({"poster_status": final_status, "poster_url": result.get("poster_url", "") if final_status == "ok" else "",
                       "poster_source": result.get("poster_source", "OMDb (IMDb ID matched)"),
                       "omdb_title": result.get("omdb_title", ""), "omdb_year": result.get("omdb_year", ""),
                       "overview": result.get("overview", ""), "poster_error": result.get("poster_error", ""),
                       "fetched_at": fetched, "error_attempts": attempt_number if final_status == "error" else 0,
                       "imdbId": linked,
                       "low_confidence": result.get("low_confidence", "false"),
                       "title_similarity": result.get("title_similarity", ""),
                       "match_reason": result.get("match_reason", "")})
        rows[mid] = record
        cache[str(mid)] = {**result, "poster_status": final_status, "fetched_at": fetched,
                           "error_attempts": attempt_number if final_status == "error" else 0,
                           "poster_url": record["poster_url"], "omdb_title": record["omdb_title"], "omdb_year": record["omdb_year"]}
        cache_file["results"] = cache
        atomic_json(CACHE, cache_file)
        saved += 1
        if saved % 25 == 0:
            write_manifest(rows)
        progress.update({"last_run_date": today, "status": "running", "resume_movieId": queue[position + 1] if position + 1 < len(queue) else None,
                         "reserved_requests_today": db.omdb_usage_count(today), "rows_saved_today": saved,
                         "queue_position": position + 1, "match_rechecks_changed": changed})
        atomic_json(PROGRESS, progress)
        print(f"OMDb {saved}: movieId={mid} status={final_status}", flush=True)
        time.sleep(args.delay)
        if stop == "network_unavailable":
            break

    write_manifest(rows)
    progress.update({"last_run_date": today, "status": stop, "resume_movieId": resume,
                     "reserved_requests_today": db.omdb_usage_count(today), "rows_saved_today": saved,
                     "match_rechecks_changed": changed})
    report = _coverage_report(movies, links, rows, progress, changed, args.daily_limit)
    previous_report = read_json(REPORT, {})
    reported_changes = max(
        changed,
        int(progress.get("match_rechecks_changed", 0) or 0),
        int(previous_report.get("rows_changed_by_current_matcher", 0) or 0),
    )
    report["rows_changed_by_current_matcher"] = reported_changes
    progress["match_rechecks_changed"] = reported_changes
    atomic_json(PROGRESS, progress)
    atomic_json(REPORT, report)
    print(f"status={stop}; budget_reservations_today={db.omdb_usage_count(today)} (started at {initial_usage}); rows_updated={saved}; resume_movieId={resume}")
    if stop == "network_unavailable":
        print("OMDb network unavailable; resume with the same command when connectivity is restored.")
    print(json.dumps(report["overall"], indent=2))
    for cutoff, summary in report["coverage_by_rating_count"].items():
        print(f"coverage rating_count>={cutoff}: {json.dumps(summary, sort_keys=True)}")
    if stop == "daily_limit_reached":
        print("resume tomorrow")
    return 0 if stop in {"complete", "daily_limit_reached"} else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-limit", type=int, default=950)
    parser.add_argument("--links", type=Path, default=DEFAULT_LINKS)
    parser.add_argument("--youtube", type=Path, default=DEFAULT_YOUTUBE)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--seed-youtube-links-only", action="store_true",
                        help="Add missing MovieLens YouTube mappings as unverified link-only candidates; do not call OMDb.")
    args = parser.parse_args()
    if args.daily_limit <= 0 or args.delay < 0:
        parser.error("daily-limit must be positive and delay must be nonnegative")
    db.initialize()
    return run_priority(args)


if __name__ == "__main__":
    raise SystemExit(main())
