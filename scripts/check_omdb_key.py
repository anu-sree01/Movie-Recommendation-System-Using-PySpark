"""Validate the OMDb key from the repository .env without displaying it."""

from pathlib import Path
import sys

import requests


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
OMDB_URL = "https://www.omdbapi.com/"


def load_key() -> str:
    if not ENV_FILE.is_file():
        return ""
    for line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if stripped.startswith("OMDB_API_KEY="):
            return stripped.partition("=")[2].strip().strip("\"'")
    return ""


def main() -> int:
    key = load_key()
    if not key:
        print("KEY EMPTY")
        return 2
    try:
        response = requests.get(
            OMDB_URL,
            params={"i": "tt0111161", "apikey": key},
            timeout=(10, 20),
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        print("CHECK FAILED: request or response error")
        return 1
    if payload.get("Title") == "The Shawshank Redemption":
        print("OK")
        return 0
    if payload.get("Error") == "Invalid API key!":
        print("KEY INVALID")
        return 2
    if "limit" in str(payload.get("Error", "")).lower():
        print("DAILY LIMIT")
        return 3
    print("CHECK FAILED: response title did not match")
    return 1


if __name__ == "__main__":
    sys.exit(main())
