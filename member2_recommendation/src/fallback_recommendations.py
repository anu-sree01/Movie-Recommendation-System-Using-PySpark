"""
Member 2 - Fallback Recommendation Generator
Module: member2_recommendation/src/fallback_recommendations.py

Generates cold-start fallback recommendations for Member 3 (Streamlit Dashboard / SQLite):
  1. Global Top 50 Popular Movies:
     - Based on Member 1's Bayesian weighted_rating with rating_count >= 1000.
     - Deterministic tie-breaker on movieId.
  2. Top 20 Genre-Specific Movies per Genre:
     - Explodes pipe-separated genres (excluding 'Unknown').
     - Ranks top 20 movies per genre using weighted_rating descending.
     - Deterministic tie-breakers on rating_count descending and movieId ascending.
"""

import os
import sys
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Add parent dir to path for imports
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from data_loader import get_default_dataset_paths, load_clean_datasets


def generate_top_popular_movies(
    movies_df: DataFrame,
    min_rating_count: int = 1000,
    top_n: int = 50,
) -> DataFrame:
    """
    Generates the Global Top-N popular movies based on Bayesian weighted_rating.
    
    Args:
        movies_df: Spark DataFrame of cleaned movies catalog.
        min_rating_count: Minimum number of ratings required (default 1000).
        top_n: Number of top movies to select (default 50).
        
    Returns:
        DataFrame containing top movies sorted by weighted_rating desc, movieId asc.
    """
    # Defensive null handling
    filtered_movies = movies_df.filter(
        (F.col("rating_count") >= min_rating_count) &
        F.col("rating_count").isNotNull() &
        F.col("weighted_rating").isNotNull()
    )

    # Sort with deterministic tie-breaker
    sorted_movies = filtered_movies.orderBy(
        F.col("weighted_rating").desc(),
        F.col("rating_count").desc(),
        F.col("movieId").asc()
    ).limit(top_n)

    # Select and format columns
    result = sorted_movies.select(
        F.col("movieId").cast("int"),
        F.col("title"),
        F.col("clean_title"),
        F.col("year"),
        F.col("genres"),
        F.col("rating_count").cast("long"),
        F.round(F.col("avg_rating"), 4).alias("avg_rating"),
        F.round(F.col("weighted_rating"), 4).alias("weighted_rating"),
    )

    return result


def generate_top_genre_movies(
    movies_df: DataFrame,
    top_n_per_genre: int = 20,
    excluded_genres: list[str] = None,
) -> DataFrame:
    """
    Generates Top-N movies per genre based on Bayesian weighted_rating.
    
    Args:
        movies_df: Spark DataFrame of cleaned movies catalog.
        top_n_per_genre: Number of recommendations per genre (default 20).
        excluded_genres: Genres to ignore (default ['Unknown', '(no genres listed)']).
        
    Returns:
        DataFrame with columns: genre, rank, movieId, title, clean_title, year,
        genres, rating_count, avg_rating, weighted_rating.
    """
    if excluded_genres is None:
        excluded_genres = ["Unknown", "(no genres listed)", ""]

    # Filter out null genres or null weighted_rating
    valid_movies = movies_df.filter(
        F.col("genres").isNotNull() &
        F.col("weighted_rating").isNotNull()
    )

    # Explode pipe-separated genres into distinct rows
    exploded_df = valid_movies.withColumn(
        "genre", F.explode(F.split(F.col("genres"), r"\|"))
    ).filter(~F.col("genre").isin(excluded_genres) & F.col("genre").isNotNull() & (F.col("genre") != ""))

    # Window specification for ranking within each genre
    genre_window = Window.partitionBy("genre").orderBy(
        F.col("weighted_rating").desc(),
        F.col("rating_count").desc(),
        F.col("movieId").asc()
    )

    ranked_df = exploded_df.withColumn(
        "rank", F.row_number().over(genre_window)
    ).filter(F.col("rank") <= top_n_per_genre)

    result = ranked_df.select(
        F.col("genre"),
        F.col("rank").cast("int"),
        F.col("movieId").cast("int"),
        F.col("title"),
        F.col("clean_title"),
        F.col("year"),
        F.col("genres"),
        F.col("rating_count").cast("long"),
        F.round(F.col("avg_rating"), 4).alias("avg_rating"),
        F.round(F.col("weighted_rating"), 4).alias("weighted_rating"),
    ).orderBy(F.col("genre").asc(), F.col("rank").asc())

    return result


def export_fallback_recommendations(
    movies_df: DataFrame,
    outputs_dir: str,
) -> tuple[str, str]:
    """
    Computes and exports both global and genre-specific fallback recommendations to CSV.
    
    Returns:
        tuple of (popular_csv_path, genre_csv_path)
    """
    os.makedirs(outputs_dir, exist_ok=True)
    popular_path = os.path.join(outputs_dir, "top_popular_movies.csv")
    genre_path = os.path.join(outputs_dir, "top_genre_movies.csv")

    print("[fallback_recommendations] Generating Global Top 50 Popular Movies (rating_count >= 1000)...")
    popular_df = generate_top_popular_movies(movies_df, min_rating_count=1000, top_n=50)
    popular_pdf = popular_df.toPandas()
    popular_pdf.to_csv(popular_path, index=False)
    print(f"[fallback_recommendations] Exported -> {popular_path} ({len(popular_pdf)} records)")

    print("[fallback_recommendations] Generating Top 20 Movies Per Genre...")
    genre_df = generate_top_genre_movies(movies_df, top_n_per_genre=20)
    genre_pdf = genre_df.toPandas()
    genre_pdf.to_csv(genre_path, index=False)
    n_genres = genre_pdf["genre"].nunique() if len(genre_pdf) > 0 else 0
    print(f"[fallback_recommendations] Exported -> {genre_path} ({len(genre_pdf)} records across {n_genres} genres)")

    return popular_path, genre_path


if __name__ == "__main__":
    from spark_session import get_spark_session, stop_spark_session

    spark, temp_dir = get_spark_session(app_name="MovieLens32M-Fallback-Generator")
    try:
        _, movies_path = get_default_dataset_paths()
        movies_df = spark.read.parquet(movies_path)
        out_dir = os.path.join(os.path.dirname(_src_dir), "outputs")
        export_fallback_recommendations(movies_df, out_dir)
    finally:
        stop_spark_session(spark, temp_dir)
