"""Load and validate the actual MovieLens and Member 2 CSV artifacts."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MOVIES_FILE = ROOT / "output" / "clean" / "movies_clean.csv"
OUTPUTS = ROOT / "member2_recommendation" / "outputs"
RECOMMENDATION_FILES = {
    "als": OUTPUTS / "pilot_user_recommendations.csv",
    "popular": OUTPUTS / "top_popular_movies.csv",
    "genre": OUTPUTS / "top_genre_movies.csv",
    "metrics": OUTPUTS / "pilot_evaluation_metrics.csv",
}
MOVIE_COLUMNS = {"movieId", "title", "genres"}
ALS_COLUMNS = {"userId", "movieId", "rank", "predicted_rating", "title", "genres"}
POPULAR_COLUMNS = {"movieId", "title", "genres", "weighted_rating"}
GENRE_COLUMNS = {"genre", "rank", "movieId", "title", "genres", "weighted_rating"}


class DataContractError(ValueError):
    """Raised when an integration CSV is missing or violates its schema."""


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise DataContractError(f"Required data file is missing: {path}")
    frame = pd.read_csv(path)
    missing = required.difference(frame.columns)
    if missing:
        raise DataContractError(f"{path.name} is missing required columns: {', '.join(sorted(missing))}")
    return frame


def load_movies(path: Path = MOVIES_FILE) -> pd.DataFrame:
    movies = _read_csv(path, MOVIE_COLUMNS)
    movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce")
    movies = movies.dropna(subset=["movieId", "title"]).copy()
    movies["movieId"] = movies["movieId"].astype("int64")
    movies["genres"] = movies["genres"].fillna("").astype(str)
    for field in ("year", "rating_count", "avg_rating", "weighted_rating"):
        if field in movies:
            movies[field] = pd.to_numeric(movies[field], errors="coerce")
    return movies.drop_duplicates("movieId", keep="first").reset_index(drop=True)


def load_als(path: Path = RECOMMENDATION_FILES["als"]) -> pd.DataFrame:
    frame = _read_csv(path, ALS_COLUMNS)
    for field in ("userId", "movieId", "rank"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame = frame.dropna(subset=["userId", "movieId", "rank"]).copy()
    frame[["userId", "movieId", "rank"]] = frame[["userId", "movieId", "rank"]].astype("int64")
    frame["predicted_rating"] = pd.to_numeric(frame["predicted_rating"], errors="coerce")
    return frame.sort_values(["userId", "rank", "movieId"], kind="stable").reset_index(drop=True)


def load_popular(path: Path = RECOMMENDATION_FILES["popular"]) -> pd.DataFrame:
    frame = _read_csv(path, POPULAR_COLUMNS)
    frame["movieId"] = pd.to_numeric(frame["movieId"], errors="coerce")
    frame["weighted_rating"] = pd.to_numeric(frame["weighted_rating"], errors="coerce")
    return frame.dropna(subset=["movieId"]).sort_values(
        ["weighted_rating", "movieId"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def load_genre_movies(path: Path = RECOMMENDATION_FILES["genre"]) -> pd.DataFrame:
    frame = _read_csv(path, GENRE_COLUMNS)
    frame["movieId"] = pd.to_numeric(frame["movieId"], errors="coerce")
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    return frame.dropna(subset=["movieId", "rank"]).sort_values(
        ["genre", "rank", "movieId"], kind="stable"
    ).reset_index(drop=True)


def load_metrics(path: Path = RECOMMENDATION_FILES["metrics"]) -> dict[str, object]:
    frame = _read_csv(path, {"rmse", "mae"})
    return frame.iloc[0].to_dict() if not frame.empty else {}
