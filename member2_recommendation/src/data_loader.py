"""
Member 2 - Dataset Ingestion & Validation
Module: member2_recommendation/src/data_loader.py
"""

import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, FloatType, LongType, StringType, DoubleType


def resolve_parquet_target(candidates: list[str]) -> str:
    """
    Resolves Parquet target path safely on Windows.
    Bypasses Hadoop NativeIO$Windows.access0 directory listing error by pointing directly
    to the underlying part file when a folder is provided.
    """
    for base in candidates:
        if os.path.isdir(base):
            parts = [f for f in os.listdir(base) if f.endswith(".parquet") and not f.startswith(".")]
            if parts:
                return os.path.join(base, parts[0])
            return base
        elif os.path.isfile(base):
            return base
    return None


def get_default_dataset_paths() -> tuple[str, str]:
    """Returns resolved paths for ratings_clean.parquet and movies_clean.parquet."""
    candidates_ratings = [
        os.path.join("output", "clean", "ratings_clean.parquet"),
        os.path.join("Movie-Recommendation-System-Using-PySpark-main", "output", "clean", "ratings_clean.parquet"),
        os.path.join("..", "output", "clean", "ratings_clean.parquet"),
    ]
    candidates_movies = [
        os.path.join("output", "clean", "movies_clean.parquet"),
        os.path.join("Movie-Recommendation-System-Using-PySpark-main", "output", "clean", "movies_clean.parquet"),
        os.path.join("..", "output", "clean", "movies_clean.parquet"),
    ]

    r_path = resolve_parquet_target(candidates_ratings)
    m_path = resolve_parquet_target(candidates_movies)

    if not r_path or not m_path:
        raise FileNotFoundError(
            f"Could not locate cleaned datasets. Checked: Ratings={r_path}, Movies={m_path}"
        )
    return r_path, m_path


def load_clean_datasets(
    spark: SparkSession,
    ratings_path: str = None,
    movies_path: str = None,
    repartition_ratings: int = None,
) -> tuple[DataFrame, DataFrame]:
    """
    Loads and validates the cleaned ratings and movies Parquet datasets.
    """
    if not ratings_path or not movies_path:
        default_r, default_m = get_default_dataset_paths()
        ratings_path = ratings_path or default_r
        movies_path = movies_path or default_m
    else:
        ratings_path = resolve_parquet_target([ratings_path]) or ratings_path
        movies_path = resolve_parquet_target([movies_path]) or movies_path

    print(f"[data_loader] Loading ratings from: {ratings_path}")
    ratings_df = spark.read.parquet(ratings_path)

    if repartition_ratings:
        print(f"[data_loader] Repartitioning ratings into {repartition_ratings} partitions...")
        ratings_df = ratings_df.repartition(repartition_ratings)

    print(f"[data_loader] Loading movies from: {movies_path}")
    movies_df = spark.read.parquet(movies_path)

    # Validate essential schema elements
    for col_name in ["userId", "movieId", "rating"]:
        if col_name not in ratings_df.columns:
            raise ValueError(f"Ratings dataset missing mandatory column: '{col_name}'")

    for col_name in ["movieId", "title", "clean_title", "genres"]:
        if col_name not in movies_df.columns:
            raise ValueError(f"Movies dataset missing mandatory column: '{col_name}'")

    return ratings_df, movies_df


def validate_data_quality(ratings_df: DataFrame) -> dict:
    """
    Checks for nulls and duplicate (userId, movieId) pairs without collecting to driver.
    """
    total_count = ratings_df.count()
    null_counts = ratings_df.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c)
        for c in ["userId", "movieId", "rating"]
    ]).first().asDict()

    distinct_pairs = ratings_df.select("userId", "movieId").distinct().count()
    duplicate_count = total_count - distinct_pairs

    return {
        "total_records": total_count,
        "null_counts": null_counts,
        "duplicate_pairs": duplicate_count,
        "partitions": ratings_df.rdd.getNumPartitions(),
    }


def split_ratings_data(
    ratings_df: DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Performs a deterministic three-way split: Train (80%), Validation (10%), Test (10%).
    """
    assert round(train_ratio + val_ratio + test_ratio, 5) == 1.0, "Split ratios must sum to 1.0"
    train_df, val_df, test_df = ratings_df.randomSplit(
        [train_ratio, val_ratio, test_ratio], seed=seed
    )
    return train_df, val_df, test_df


def get_pilot_sample(ratings_df: DataFrame, fraction: float = 0.005, seed: int = 42) -> DataFrame:
    """
    Extracts a representative sample fraction of ratings for pilot validation.
    Default 0.005 (0.5%) extracts ~160,000 ratings from 32M records.
    """
    return ratings_df.sample(withReplacement=False, fraction=fraction, seed=seed)
