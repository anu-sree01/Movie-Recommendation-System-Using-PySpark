"""
Member 2 - Pilot Execution Pipeline
Module: member2_recommendation/src/run_pilot.py

Executes a controlled, resource-safe pilot experiment:
  1. Starts a local SparkSession with defensive memory allocation.
  2. Ingests cleaned MovieLens 32M Parquet datasets via Windows-safe target resolver.
  3. Samples a representative 0.5% subset (~160k ratings) to protect host RAM.
  4. Performs a deterministic 80/10/10 split (Train/Val/Test).
  5. Fits an ALS collaborative filtering model (rank=10, regParam=0.1, maxIter=5).
  6. Evaluates predictions on the Validation set (RMSE and MAE).
  7. Serializes the pilot model factor matrices into models/als_pilot_model/.
  8. Generates Top-10 personalized movie recommendations for 50 sample users,
     strictly excluding previously rated training movies and clipping display ratings.
  9. Enriches recommendations with movie metadata and exports to outputs/.
 10. Generates cold-start fallback datasets (Global Top 50 and Genre-Specific Top 20).
"""

import os
import sys
import time

# Ensure src/ package imports work smoothly
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from spark_session import get_spark_session, stop_spark_session
from data_loader import (
    load_clean_datasets,
    validate_data_quality,
    split_ratings_data,
    get_pilot_sample,
)
from train_als import build_als_estimator, fit_als_model, save_als_model
from evaluate_model import evaluate_predictions, save_evaluation_metrics
from generate_recommendations import recommend_for_users, enrich_recommendations, export_recommendations_csv
from fallback_recommendations import export_fallback_recommendations

from pyspark import StorageLevel
from pyspark.sql import functions as F


def banner(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def run_pilot_pipeline():
    start_total_time = time.time()
    banner("MEMBER 2: ALS RECOMMENDATION ENGINE — CONTROLLED PILOT EXECUTION")

    # 1. Initialize SparkSession
    banner("STEP 1: SPARK INITIALIZATION")
    spark, temp_dir = get_spark_session(
        app_name="MovieLens32M-ALS-Pilot",
        driver_memory="3g",
        shuffle_partitions=32,
    )
    print(f"Spark Version        : {spark.version}")
    print(f"Staging Directory    : {temp_dir}")

    # 2. Ingest Cleaned Datasets
    banner("STEP 2: DATASET INGESTION & QUALITY AUDIT")
    ratings_full, movies_df = load_clean_datasets(spark)
    print("Performing data quality check on ratings...")
    quality_stats = validate_data_quality(ratings_full)
    print(f"Full Ratings Records : {quality_stats['total_records']:,}")
    print(f"Null Counts          : {quality_stats['null_counts']}")
    print(f"Duplicate Pairs      : {quality_stats['duplicate_pairs']}")
    print(f"Partition Count      : {quality_stats['partitions']}")
    print(f"Total Movies Catalog : {movies_df.count():,}")

    # 3. Controlled Pilot Subsampling (0.5% fraction = ~160k ratings)
    banner("STEP 3: PILOT SUBSAMPLING & THREE-WAY SPLIT")
    sample_fraction = 0.005  # 0.5% representative sample
    print(f"Sampling {sample_fraction * 100:.1f}% representative subset for pilot testing...")
    ratings_sample = get_pilot_sample(ratings_full, fraction=sample_fraction, seed=42)

    # Three-way split: 80% train, 10% validation, 10% test
    train_df, val_df, test_df = split_ratings_data(
        ratings_sample, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
    )

    # Persist train and val in memory and disk for ALS iterative computation
    train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)
    val_df   = val_df.persist(StorageLevel.MEMORY_AND_DISK)

    n_train = train_df.count()
    n_val   = val_df.count()
    n_test  = test_df.count()
    total_pilot = n_train + n_val + n_test

    print(f"Pilot Dataset Breakdown:")
    print(f"  • Training Set   (80%): {n_train:,} records")
    print(f"  • Validation Set (10%): {n_val:,} records")
    print(f"  • Test Set       (10%): {n_test:,} records (Held out / untouched)")
    print(f"  • Total Pilot Sample  : {total_pilot:,} records")

    # 4. ALS Model Training
    banner("STEP 4: ALS MODEL FITTING (PILOT CONFIGURATION)")
    pilot_config = {
        "rank": 10,
        "regParam": 0.1,
        "maxIter": 5,
        "seed": 42,
        "coldStartStrategy": "drop",
        "nonnegative": True,
        "implicitPrefs": False,
    }
    print(f"Pilot Hyperparameters: {pilot_config}")

    als_estimator = build_als_estimator(**pilot_config)
    model, train_elapsed = fit_als_model(als_estimator, train_df)

    # 5. Validation Evaluation
    banner("STEP 5: MODEL EVALUATION ON VALIDATION SET")
    metrics, val_predictions = evaluate_predictions(
        model=model,
        eval_df=val_df,
        dataset_name="Validation Set (Pilot)",
        cold_start_strategy="drop",
    )
    metrics["training_time_seconds"] = round(train_elapsed, 2)
    metrics["hyperparameters"] = pilot_config
    metrics["pilot_sample_fraction"] = sample_fraction
    metrics["pilot_total_records"] = total_pilot
    metrics["train_records"] = n_train

    # Determine paths
    repo_root = os.path.dirname(_src_dir)
    outputs_dir = os.path.join(repo_root, "outputs")
    models_dir  = os.path.join(repo_root, "models")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(models_dir,  exist_ok=True)

    metrics_json = os.path.join(outputs_dir, "pilot_evaluation_metrics.json")
    metrics_csv  = os.path.join(outputs_dir, "pilot_evaluation_metrics.csv")
    save_evaluation_metrics(metrics, metrics_json, metrics_csv)

    # 6. Model Serialization
    banner("STEP 6: SERIALIZING PILOT MODEL")
    model_save_path = os.path.join(models_dir, "als_pilot_model")
    save_als_model(model, model_save_path, temp_dir=temp_dir, overwrite=True)

    # 7. Recommendation Generation (Expanded to 50 users with exclusion of rated movies)
    banner("STEP 7: GENERATING TOP-10 RECOMMENDATIONS FOR 50 SAMPLE USERS")
    # Select 50 active users who have rating history in train_df
    sample_users = (
        val_df.select("userId")
        .distinct()
        .join(train_df.select("userId").distinct(), on="userId", how="inner")
        .limit(50)
    )
    n_sample_users = sample_users.count()
    print(f"Selected {n_sample_users} sample users for recommendation generation.")

    raw_recs = recommend_for_users(
        model=model,
        users_df=sample_users,
        train_ratings_df=train_df,
        num_recommendations=10,
        candidate_pool_size=100,
    )
    enriched_recs = enrich_recommendations(raw_recs, movies_df)

    recs_csv_path = os.path.join(outputs_dir, "pilot_user_recommendations.csv")
    export_recommendations_csv(enriched_recs, recs_csv_path)

    print("\nSample Enriched Recommendations Preview (First 15 rows):")
    enriched_recs.select(
        "userId", "rank", "movieId", "clean_title", "genres", "raw_prediction", "predicted_rating"
    ).show(15, truncate=False)

    # 8. Cold-Start Fallback Recommendations (Global Top 50 & Genre Top 20)
    banner("STEP 8: GENERATING COLD-START FALLBACK DATASETS FOR MEMBER 3")
    export_fallback_recommendations(movies_df, outputs_dir)

    # 9. Clean up cached data
    train_df.unpersist()
    val_df.unpersist()

    total_elapsed = time.time() - start_total_time
    banner(f"PILOT PIPELINE COMPLETED SUCCESSFULLY IN {total_elapsed:.2f} SECONDS")

    stop_spark_session(spark, temp_dir)


if __name__ == "__main__":
    run_pilot_pipeline()
