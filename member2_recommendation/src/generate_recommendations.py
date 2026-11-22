"""
Member 2 - Top-N Recommendation Generation & Metadata Enrichment
Module: member2_recommendation/src/generate_recommendations.py

Generates personalized Top-N movie recommendations with:
  1. Anti-join exclusion of previously rated movies from the user's training history.
  2. Large candidate generation pool (e.g., 100 items per user) to guarantee enough unrated items.
  3. Continuous ranking (1..10) per user without ties or gaps.
  4. Dual-prediction handling:
     - raw_prediction: unclipped ALS factor dot product (rounded to 3 decimals).
     - predicted_rating: display rating clamped to the valid [0.5, 5.0] star scale.
  5. Movie metadata enrichment (title, clean_title, year, genres, ratings summary).
  6. Direct factor-based inference option (recommend_for_users_from_factors) for Member 3's
     Streamlit dashboard without requiring an active Spark cluster.
"""

import os
import sys
import numpy as np
import pandas as pd
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def recommend_for_users(
    model: ALSModel,
    users_df: DataFrame,
    train_ratings_df: DataFrame = None,
    num_recommendations: int = 10,
    candidate_pool_size: int = 100,
) -> DataFrame:
    """
    Generates Top-N movie recommendations for a given subset of users using PySpark ALSModel,
    strictly excluding movies that each user has already rated in train_ratings_df.
    
    Args:
        model: Trained ALSModel.
        users_df: DataFrame containing at least a 'userId' column.
        train_ratings_df: Optional DataFrame of historical ratings (userId, movieId) to exclude.
        num_recommendations: Target number of recommendations per user (default 10).
        candidate_pool_size: Number of initial candidates generated per user (default 100).
        
    Returns:
        DataFrame with columns: userId, rank, movieId, raw_prediction, predicted_rating
    """
    user_subset = users_df.select("userId").distinct()

    # 1. Generate candidate pool from ALS model (oversampling to allow for rated movie exclusion)
    raw_recs = model.recommendForUserSubset(user_subset, candidate_pool_size)

    # 2. Explode the array of struct<movieId:int, rating:float>
    exploded_recs = raw_recs.select(
        F.col("userId"),
        F.explode(F.col("recommendations")).alias("rec")
    ).select(
        F.col("userId").cast("int"),
        F.col("rec.movieId").cast("int").alias("movieId"),
        F.col("rec.rating").alias("raw_score")
    )

    # 3. Exclude previously rated movies using left_anti join
    if train_ratings_df is not None:
        user_history = train_ratings_df.select(
            F.col("userId").cast("int"),
            F.col("movieId").cast("int")
        ).distinct()
        candidates = exploded_recs.join(
            user_history,
            on=["userId", "movieId"],
            how="left_anti"
        )
    else:
        candidates = exploded_recs

    # 4. Rank unrated candidates per user with deterministic tie-breaking
    window_spec = Window.partitionBy("userId").orderBy(
        F.col("raw_score").desc(),
        F.col("movieId").asc()
    )

    ranked_candidates = candidates.withColumn(
        "rank", F.row_number().over(window_spec)
    ).filter(F.col("rank") <= num_recommendations)

    # 5. Format prediction columns:
    # - raw_prediction: original model output (rounded to 3 decimals)
    # - predicted_rating: display score clamped to [0.5, 5.0] (rounded to 3 decimals)
    formatted_recs = ranked_candidates.select(
        F.col("userId"),
        F.col("rank").cast("int"),
        F.col("movieId").cast("int"),
        F.round(F.col("raw_score"), 3).alias("raw_prediction"),
        F.round(
            F.when(F.col("raw_score") > 5.0, 5.0)
             .when(F.col("raw_score") < 0.5, 0.5)
             .otherwise(F.col("raw_score")),
            3
        ).alias("predicted_rating"),
    )

    return formatted_recs


def enrich_recommendations(
    recommendations_df: DataFrame,
    movies_df: DataFrame,
) -> DataFrame:
    """
    Joins recommended movie IDs with movie metadata for human-readable output.
    """
    movie_metadata = movies_df.select(
        "movieId", "title", "clean_title", "year", "genres",
        "rating_count", "avg_rating", "weighted_rating"
    )

    enriched = (
        recommendations_df
        .join(movie_metadata, on="movieId", how="left")
        .select(
            "userId",
            "rank",
            "movieId",
            "title",
            "clean_title",
            "year",
            "genres",
            "raw_prediction",
            "predicted_rating",
            "rating_count",
            "avg_rating",
            "weighted_rating",
        )
        .orderBy("userId", "rank")
    )
    return enriched


def export_recommendations_csv(enriched_df: DataFrame, output_csv_path: str) -> None:
    """Saves enriched recommendations as a clean CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    pdf = enriched_df.toPandas()
    pdf.to_csv(output_csv_path, index=False)
    print(f"[generate_recommendations] Exported recommendations -> {output_csv_path} ({len(pdf)} rows)")


def recommend_for_users_from_factors(
    user_factors_df: pd.DataFrame,
    item_factors_df: pd.DataFrame,
    user_ids: list[int],
    train_ratings_df: pd.DataFrame = None,
    movies_df: pd.DataFrame = None,
    num_recommendations: int = 10,
) -> pd.DataFrame:
    """
    Computes personalized recommendations directly from factor matrices using NumPy dot products.
    Ideal for Member 3's Streamlit dashboard (zero Spark cluster overhead).
    
    Args:
        user_factors_df: DataFrame with 'id' and 'features' (list of floats).
        item_factors_df: DataFrame with 'id' and 'features' (list of floats).
        user_ids: List of target user IDs to recommend for.
        train_ratings_df: Optional DataFrame with 'userId' and 'movieId' to exclude already-rated movies.
        movies_df: Optional DataFrame of movies metadata for enrichment.
        num_recommendations: Number of recommendations per user (default 10).
        
    Returns:
        pandas DataFrame of enriched recommendations.
    """
    # Build item matrix: shape (N_items, rank)
    item_ids = item_factors_df["id"].values
    item_matrix = np.vstack(item_factors_df["features"].values)  # (N_items, rank)

    # Filter to requested users present in user_factors
    available_users = set(user_factors_df["id"].values)
    target_users = [u for u in user_ids if u in available_users]

    if not target_users:
        print("[recommend_for_users_from_factors] None of the requested user IDs found in user_factors.")
        return pd.DataFrame()

    user_subset_df = user_factors_df[user_factors_df["id"].isin(target_users)]
    user_matrix = np.vstack(user_subset_df["features"].values)  # (N_target_users, rank)
    user_id_list = user_subset_df["id"].values

    # Pre-index historical ratings if provided
    history_dict = {}
    if train_ratings_df is not None and len(train_ratings_df) > 0:
        grouped = train_ratings_df[train_ratings_df["userId"].isin(target_users)].groupby("userId")["movieId"].apply(set)
        history_dict = grouped.to_dict()

    # Pre-index movies metadata if provided
    movies_lookup = {}
    if movies_df is not None:
        movies_lookup = movies_df.set_index("movieId").to_dict(orient="index")

    rows = []
    # Compute dot products: (N_target_users, N_items)
    score_matrix = np.dot(user_matrix, item_matrix.T)

    for u_idx, uid in enumerate(user_id_list):
        scores = score_matrix[u_idx]  # shape (N_items,)
        rated_set = history_dict.get(uid, set())

        # Mask out previously rated movies
        if rated_set:
            mask = np.isin(item_ids, list(rated_set))
            scores = scores.copy()
            scores[mask] = -1e9  # effectively remove

        # Find top N indices
        top_indices = np.argsort(scores)[::-1][:num_recommendations]

        rank = 1
        for idx in top_indices:
            raw_score = float(scores[idx])
            if raw_score <= -1e8:
                break  # Exhausted unrated movies
            mid = int(item_ids[idx])
            raw_pred = round(raw_score, 3)
            display_pred = round(max(0.5, min(5.0, raw_score)), 3)

            meta = movies_lookup.get(mid, {})
            row = {
                "userId": int(uid),
                "rank": rank,
                "movieId": mid,
                "title": meta.get("title", ""),
                "clean_title": meta.get("clean_title", ""),
                "year": meta.get("year", None),
                "genres": meta.get("genres", ""),
                "raw_prediction": raw_pred,
                "predicted_rating": display_pred,
                "rating_count": meta.get("rating_count", 0),
                "avg_rating": round(meta.get("avg_rating", 0.0), 4) if meta.get("avg_rating") is not None else None,
                "weighted_rating": round(meta.get("weighted_rating", 0.0), 4) if meta.get("weighted_rating") is not None else None,
            }
            rows.append(row)
            rank += 1

    return pd.DataFrame(rows)
