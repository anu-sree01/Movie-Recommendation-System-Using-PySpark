"""Shared title and release-year sanity matching for OMDb movie records."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


ARTICLES = {"the", "a", "an", "les", "le", "la", "el", "das", "der", "die"}


def normalize_catalog_title(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii").casefold()
    text = re.sub(r"\s*\((?:19|20)\d{2}\)\s*$", "", text)
    text = re.sub(r"\b(?:19|20)\d{2}\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    variants = [re.sub(r"\s*\([^()]*\)", " ", text), *re.findall(r"\(([^()]*)\)", text)]
    normalized = []
    for variant in variants:
        variant = re.sub(r"\s+", " ", variant).strip()
        match = re.search(r",\s*([a-z]+)$", variant)
        if match and match.group(1) in ARTICLES:
            variant = f"{match.group(1)} {variant[:match.start()].strip()}"
        result = " ".join(re.findall(r"[a-z0-9]+", variant))
        if result:
            normalized.append(result)
    return "\n".join(normalized)


def title_similarity(left: str, right: str) -> float:
    left_variants = normalize_catalog_title(left).splitlines()
    right_variants = normalize_catalog_title(right).splitlines()
    if not left_variants or not right_variants:
        return 0.0
    return max(SequenceMatcher(None, a, b).ratio() for a in left_variants for b in right_variants)


def match_omdb_record(movie_title: str, movie_year, omdb_title: str, omdb_year) -> tuple[bool, bool, float, str]:
    """Return accepted, low-confidence, title similarity, and a diagnostic reason."""
    similarity = title_similarity(movie_title, omdb_title)
    if similarity < 0.8:
        return False, False, similarity, "title_similarity_below_0.8"
    try:
        local_year = int(float(movie_year))
    except (TypeError, ValueError, OverflowError):
        local_year = None
    if local_year is None:
        return True, True, similarity, "catalogue_year_missing"
    years = [int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", str(omdb_year))]
    if not years or min(abs(local_year - value) for value in years) > 1:
        return False, False, similarity, "year_outside_tolerance"
    return True, False, similarity, "matched"
