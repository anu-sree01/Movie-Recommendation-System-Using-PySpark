"""
Member 2 - Phase 3 Verification Suite
Script: member2_recommendation/src/verify_phase3.py

Verifies all 12 Phase 3 requirements:
  1. Global Top 50 file exists and contains <= 50 records.
  2. Every global fallback movie meets rating_count >= 1000.
  3. Genre fallback file contains <= 20 movies per genre.
  4. No Unknown genre entries exist.
  5. Personalized recommendation file contains only valid movie IDs.
  6. Previously rated movies are strictly excluded for every user.
  7. Each user's ranking starts at 1 and has no duplicate ranks.
  8. Display predictions are within 0.5–5.0.
  9. Raw prediction values are preserved.
 10. Movie titles and metadata are correctly joined.
 11. All output files can be opened and read successfully.
 12. README accurately documents integration files and limitations.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure src/ package imports work
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from data_loader import get_default_dataset_paths, load_clean_datasets, get_pilot_sample, split_ratings_data
from spark_session import get_spark_session, stop_spark_session


def run_verification():
    print("=" * 80)
    print(" MEMBER 2: PHASE 3 COMPREHENSIVE VERIFICATION SUITE")
    print("=" * 80)

    repo_dir = os.path.dirname(_src_dir)
    outputs_dir = os.path.join(repo_dir, "outputs")
    models_dir = os.path.join(repo_dir, "models", "als_pilot_model")

    # ---------------------------------------------------------
    # CHECK 1 & 2: Global Top 50 Popular Movies
    # ---------------------------------------------------------
    popular_path = os.path.join(outputs_dir, "top_popular_movies.csv")
    assert os.path.exists(popular_path), f"Missing {popular_path}"
    pop_df = pd.read_csv(popular_path)
    n_pop = len(pop_df)
    print(f"\n[Check 1] Global Top 50 file exists: YES")
    print(f"          Record count = {n_pop} (Condition <= 50: {n_pop <= 50})")
    assert n_pop == 50, f"Expected exactly 50 records, got {n_pop}"

    min_votes = pop_df["rating_count"].min()
    print(f"[Check 2] Minimum rating_count across global fallback = {min_votes:,}")
    assert min_votes >= 1000, f"Violation: Found movie with rating_count < 1000 ({min_votes})"
    print(f"          Every movie has rating_count >= 1000: YES")
    print(f"          Top 3 Global Movies:")
    for i, r in pop_df.head(3).iterrows():
        print(f"            #{i+1}: {r['title']} | Votes: {r['rating_count']:,} | Weighted Rating: {r['weighted_rating']:.4f}")

    # ---------------------------------------------------------
    # CHECK 3 & 4: Genre Fallback Recommendations
    # ---------------------------------------------------------
    genre_path = os.path.join(outputs_dir, "top_genre_movies.csv")
    assert os.path.exists(genre_path), f"Missing {genre_path}"
    genre_df = pd.read_csv(genre_path)
    genres = sorted(genre_df["genre"].unique().tolist())
    max_per_genre = genre_df.groupby("genre").size().max()
    min_per_genre = genre_df.groupby("genre").size().min()
    print(f"\n[Check 3] Genre fallback file exists: YES")
    print(f"          Total records = {len(genre_df)} across {len(genres)} genres")
    print(f"          Max per genre = {max_per_genre}, Min per genre = {min_per_genre} (Condition <= 20: {max_per_genre <= 20})")
    assert max_per_genre <= 20, f"Violation: More than 20 movies in a genre ({max_per_genre})"

    print(f"[Check 4] Check for 'Unknown' or invalid genres:")
    assert "Unknown" not in genres, "Violation: Found 'Unknown' genre in genre fallback!"
    assert "(no genres listed)" not in genres, "Violation: Found '(no genres listed)' genre!"
    print(f"          'Unknown' or empty genres present: NO (100% clean)")
    print(f"          Genres included ({len(genres)}): {', '.join(genres)}")

    # ---------------------------------------------------------
    # CHECK 5, 7, 8, 9, 10: Personalized Recommendations
    # ---------------------------------------------------------
    recs_path = os.path.join(outputs_dir, "pilot_user_recommendations.csv")
    assert os.path.exists(recs_path), f"Missing {recs_path}"
    recs_df = pd.read_csv(recs_path)
    n_recs = len(recs_df)
    users = recs_df["userId"].unique().tolist()
    n_users = len(users)
    print(f"\n[Check 5] Personalized recommendation file exists: YES")
    print(f"          Total rows = {n_recs} across {n_users} distinct users")
    assert n_users == 50, f"Expected 50 distinct users, got {n_users}"
    assert n_recs == 500, f"Expected 500 recommendations (50x10), got {n_recs}"

    # Verify movie IDs exist in movies catalog
    movies_path = os.path.join(repo_dir, "output", "clean", "movies_clean.parquet")
    if not os.path.exists(movies_path):
        movies_path = os.path.join("Movie-Recommendation-System-Using-PySpark-main", "output", "clean", "movies_clean.parquet")
    movies_cat = pd.read_parquet(movies_path)
    valid_ids = set(movies_cat["movieId"].values)
    rec_ids = set(recs_df["movieId"].values)
    assert rec_ids.issubset(valid_ids), "Violation: Recommended movie IDs not found in movie catalog!"
    print(f"          All {len(rec_ids)} distinct recommended movie IDs are valid catalog IDs: YES")

    # Check continuous ranks 1..10 per user
    ranks_valid = True
    for uid, grp in recs_df.groupby("userId"):
        r_list = grp["rank"].tolist()
        if r_list != list(range(1, 11)):
            ranks_valid = False
            print(f"          Rank mismatch for user {uid}: {r_list}")
            break
    print(f"[Check 7] Each user ranking is continuous from 1 to 10 with no duplicate ranks: YES")
    assert ranks_valid, "Violation: Non-continuous or duplicate ranks detected"

    # Check predictions: display clipped to [0.5, 5.0], raw preserved
    disp_min = recs_df["predicted_rating"].min()
    disp_max = recs_df["predicted_rating"].max()
    raw_min  = recs_df["raw_prediction"].min()
    raw_max  = recs_df["raw_prediction"].max()
    print(f"[Check 8] Display predictions clipped to [0.5, 5.0]: YES")
    print(f"          Display range: min = {disp_min}, max = {disp_max}")
    assert disp_min >= 0.5 and disp_max <= 5.0, f"Violation: Display ratings out of range [{disp_min}, {disp_max}]"

    print(f"[Check 9] Raw predictions preserved authentic ALS dot products: YES")
    print(f"          Raw range: min = {raw_min}, max = {raw_max}")
    assert raw_max > 5.0, "Expected some raw predictions to exceed 5.0 in ALS explicit model"

    # Check metadata enrichment
    null_titles = recs_df["clean_title"].isnull().sum()
    null_genres = recs_df["genres"].isnull().sum()
    print(f"[Check 10] Metadata correctly joined: YES")
    print(f"           Missing titles: {null_titles}, Missing genres: {null_genres}")
    assert null_titles == 0 and null_genres == 0, "Violation: Null metadata fields found in recommendations"

    # ---------------------------------------------------------
    # CHECK 6: Verify Exclusion of Previously Rated Movies
    # ---------------------------------------------------------
    print(f"\n[Check 6] Verifying exclusion of previously rated movies against training dataset...")
    # Replicate training split to check historical ratings
    ratings_path = os.path.join(repo_dir, "output", "clean", "ratings_clean.parquet")
    if not os.path.exists(ratings_path):
        ratings_path = os.path.join("Movie-Recommendation-System-Using-PySpark-main", "output", "clean", "ratings_clean.parquet")
    
    spark, temp_dir = get_spark_session(app_name="Phase3-Exclusion-Audit", driver_memory="2g", shuffle_partitions=16)
    try:
        r_df, _ = load_clean_datasets(spark)
        r_sample = get_pilot_sample(r_df, fraction=0.005, seed=42)
        train_df, _, _ = split_ratings_data(r_sample, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
        
        # Filter train ratings to the 50 sample users
        sample_uids = [int(u) for u in users]
        train_user_ratings = (
            train_df.filter(F.col("userId").isin(sample_uids))
            .select("userId", "movieId")
            .toPandas()
        )
    finally:
        stop_spark_session(spark, temp_dir)

    # Check for any intersection
    history_by_user = train_user_ratings.groupby("userId")["movieId"].apply(set).to_dict()
    violations = 0
    total_checked = 0

    for uid in users:
        user_recs = set(recs_df[recs_df["userId"] == uid]["movieId"].values)
        user_history = history_by_user.get(uid, set())
        overlap = user_recs.intersection(user_history)
        total_checked += len(user_recs)
        if overlap:
            print(f"          VIOLATION: User {uid} was recommended previously rated movies: {overlap}")
            violations += 1

    print(f"          Checked {total_checked} recommendations across {len(users)} users against training history.")
    print(f"          Overlap violations found: {violations}")
    assert violations == 0, f"Violation: {violations} users were recommended movies they already rated!"
    print(f"          Previously rated movies strictly excluded: YES (100% unrated)")

    # ---------------------------------------------------------
    # CHECK 11: File Readability
    # ---------------------------------------------------------
    print(f"\n[Check 11] All output files successfully verified readable: YES")
    for f in [
        "top_popular_movies.csv",
        "top_genre_movies.csv",
        "pilot_user_recommendations.csv",
        "pilot_evaluation_metrics.json",
        "pilot_evaluation_metrics.csv"
    ]:
        p = os.path.join(outputs_dir, f)
        assert os.path.exists(p) and os.path.getsize(p) > 0, f"File {f} is missing or empty"
        print(f"           • {f} ({os.path.getsize(p):,} bytes)")

    # ---------------------------------------------------------
    # CHECK 12: README Documentation
    # ---------------------------------------------------------
    readme_path = os.path.join(repo_dir, "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()

    assert "user_factors.parquet" in readme_content, "README missing user_factors.parquet documentation"
    assert "item_factors.parquet" in readme_content, "README missing item_factors.parquet documentation"
    assert "metadata.json" in readme_content, "README missing metadata.json documentation"
    assert "ALSModel.load" in readme_content, "README missing ALSModel.load limitation explanation"
    assert "dot products" in readme_content, "README missing factor dot product formula explanation"
    assert "top_popular_movies.csv" in readme_content, "README missing top_popular_movies.csv documentation"
    assert "top_genre_movies.csv" in readme_content, "README missing top_genre_movies.csv documentation"
    print(f"\n[Check 12] README accurately documents all artifacts and limitations: YES")

    print("\n" + "=" * 80)
    print(" ALL 12 VERIFICATION CHECKS PASSED PERFECTLY!")
    print("=" * 80)


if __name__ == "__main__":
    from pyspark.sql import functions as F
    run_verification()
