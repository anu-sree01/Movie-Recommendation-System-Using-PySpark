"""Keep ALS, popular and genre discovery results clearly separated."""

import pandas as pd


def pilot_user_ids(als: pd.DataFrame) -> list[int]:
    if als.empty:
        return []
    return sorted(als["userId"].dropna().astype(int).unique().tolist())


def personalized(als: pd.DataFrame, movielens_user_id: int, hidden_movie_ids=()) -> pd.DataFrame:
    rows = als.loc[als["userId"].eq(int(movielens_user_id))].copy()
    if hidden_movie_ids:
        rows = rows.loc[~rows["movieId"].isin(set(map(int, hidden_movie_ids)))]
    return rows.sort_values(["rank", "movieId"], kind="stable").head(10).reset_index(drop=True)


def genre_discovery(genre_rows: pd.DataFrame, genres: list[str], hidden_movie_ids=()) -> pd.DataFrame:
    if not genres or genre_rows.empty:
        return genre_rows.iloc[0:0].copy()
    selected = genre_rows.loc[genre_rows["genre"].isin(genres)].copy()
    if hidden_movie_ids:
        selected = selected.loc[~selected["movieId"].isin(set(map(int, hidden_movie_ids)))]
    return selected.sort_values(["rank", "weighted_rating", "movieId"], ascending=[True, False, True], kind="stable").drop_duplicates("movieId").head(30).reset_index(drop=True)


def popular_fallback(popular: pd.DataFrame, hidden_movie_ids=()) -> pd.DataFrame:
    rows = popular.copy()
    if hidden_movie_ids:
        rows = rows.loc[~rows["movieId"].isin(set(map(int, hidden_movie_ids)))]
    return rows.head(30).reset_index(drop=True)
