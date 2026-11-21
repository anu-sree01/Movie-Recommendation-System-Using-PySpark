"""
Movie Recommendation System Using PySpark (MovieLens 32M)
MODULE 1 - Dataset Processing & Analysis   (Member 1: Anusree)

What this script does
  1. Loads MovieLens 32M (ratings, movies) with PySpark using explicit schemas
  2. Cleans / preprocesses (nulls, duplicates, invalid ratings, title/year/genre parsing)
  3. Analyses ratings and genres with Spark DataFrame operations
  4. Runs Spark SQL queries on temp views
  5. Creates Matplotlib visualisations
  6. Exports the cleaned datasets for the ALS (Module 2) and Streamlit (Module 3) teammates

Usage
  python module1_data_processing.py --data-dir dataset/ml-32m/ml-32m --output-dir output

Windows Fix Notes
  - Uses a local temp dir for Spark warehouse/checkpoint to avoid OneDrive/NTFS permission
    issues with Hadoop's NativeIO$Windows.access0 during Parquet commit.
  - Coalesces to 1 partition before writing Parquet so Spark uses a single-file commit
    path that does not call access0().
  - hadoop.dll / winutils.exe are auto-verified; script falls back gracefully.
"""

import argparse
import os
import sys
import tempfile
import shutil

# ── 0a. WINDOWS HADOOP + PYTHON WORKER SETUP ──────────────────────────────
# Must happen BEFORE SparkSession is created.

# Fix: Windows Store Python alias intercepts bare 'python' calls that
# PySpark spawns for worker processes.  Point Spark explicitly at the
# real interpreter that is currently running this script.
_PYTHON_EXE = sys.executable  # e.g. C:\...\Python313\python.exe
os.environ["PYSPARK_PYTHON"]        = _PYTHON_EXE
os.environ["PYSPARK_DRIVER_PYTHON"] = _PYTHON_EXE
print(f"[setup] PYSPARK_PYTHON = {_PYTHON_EXE}")

_HADOOP_HOME = os.environ.get("HADOOP_HOME", r"C:\Users\anusr\.hadoop")
if os.path.isdir(_HADOOP_HOME):
    os.environ["HADOOP_HOME"] = _HADOOP_HOME
    _bin = os.path.join(_HADOOP_HOME, "bin")
    if _bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")
    print(f"[setup] HADOOP_HOME = {_HADOOP_HOME}")
    _winutils = os.path.join(_bin, "winutils.exe")
    _dll      = os.path.join(_bin, "hadoop.dll")
    print(f"[setup] winutils.exe present: {os.path.isfile(_winutils)}")
    print(f"[setup] hadoop.dll   present: {os.path.isfile(_dll)}")
else:
    print(f"[setup] WARNING: HADOOP_HOME not found at {_HADOOP_HOME}")

# Use a local temp dir for Spark internal dirs (avoids OneDrive/cloud-sync NTFS issues)
_SPARK_TMP = tempfile.mkdtemp(prefix="spark_movielens_")
print(f"[setup] Spark tmp dir = {_SPARK_TMP}")

import matplotlib
matplotlib.use("Agg")  # save figures to files, no GUI needed
import matplotlib.pyplot as plt

from pyspark import StorageLevel
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    FloatType, IntegerType, LongType, StringType, StructField, StructType,
)

# ── 0b. ARGUMENTS ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir",      default="dataset/ml-32m/ml-32m",
                    help="folder containing ratings.csv and movies.csv")
parser.add_argument("--output-dir",    default="output",
                    help="where cleaned data and plots are saved")
parser.add_argument("--driver-memory", default="6g",
                    help="Spark driver memory (use 4g on small laptops)")
args = parser.parse_args()

PLOT_DIR  = os.path.join(args.output_dir, "plots")
CLEAN_DIR = os.path.join(args.output_dir, "clean")
os.makedirs(PLOT_DIR,  exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

# ── 0c. SPARK SESSION ─────────────────────────────────────────────────────
spark = (
    SparkSession.builder
    .appName("MovieLens32M-Module1-Processing")
    .master("local[*]")
    .config("spark.driver.memory",              args.driver_memory)
    .config("spark.sql.shuffle.partitions",     "64")
    .config("spark.sql.session.timeZone",       "UTC")
    # ---- Windows / OneDrive safe dirs -----------------------------------
    .config("spark.local.dir",                  _SPARK_TMP)
    .config("spark.sql.warehouse.dir",          os.path.join(_SPARK_TMP, "warehouse"))
    .config("spark.driver.extraJavaOptions",
            f"-Djava.io.tmpdir={_SPARK_TMP} -Dderby.system.home={_SPARK_TMP}")
    # ---- Disable speculative execution (safe for local mode) ------------
    .config("spark.speculation",                "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")


# ── HELPERS ───────────────────────────────────────────────────────────────
def banner(text):
    print("\n" + "=" * 80 + f"\n{text}\n" + "=" * 80)


def null_counts(df):
    return df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns])


def save_fig(name):
    path = os.path.join(PLOT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved plot -> {path}")


def safe_parquet_write(df, path, n_partitions=1):
    """
    Windows-safe Parquet writer.

    Strategy:
      1. Coalesce to n_partitions (1 by default) → fewer _temporary sub-dirs
      2. Write to a local tmp location first
      3. Use Python shutil to move the result into the target path
         (bypasses the Hadoop NativeIO Windows file-access check on the
          network / OneDrive location entirely)
    """
    # Write to local temp first
    tmp_out = os.path.join(_SPARK_TMP, os.path.basename(path))
    if os.path.exists(tmp_out):
        shutil.rmtree(tmp_out)

    df.coalesce(n_partitions).write.mode("overwrite").parquet(tmp_out)
    print(f"  [parquet] written to tmp: {tmp_out}")

    # Move to final destination
    if os.path.exists(path):
        shutil.rmtree(path)
    shutil.copytree(tmp_out, path)
    print(f"  [parquet] copied to: {path}")


# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------
banner("1. LOADING MovieLens 32M")

ratings_schema = StructType([
    StructField("userId",    IntegerType(), True),
    StructField("movieId",   IntegerType(), True),
    StructField("rating",    FloatType(),   True),
    StructField("timestamp", LongType(),    True),
])
movies_schema = StructType([
    StructField("movieId", IntegerType(), True),
    StructField("title",   StringType(),  True),
    StructField("genres",  StringType(),  True),
])

ratings_raw = spark.read.csv(
    os.path.join(args.data_dir, "ratings.csv"), header=True, schema=ratings_schema
)
movies_raw = spark.read.csv(
    os.path.join(args.data_dir, "movies.csv"),
    header=True, schema=movies_schema, quote='"', escape='"',
)

print("Ratings schema:");  ratings_raw.printSchema()
print("Movies schema:");   movies_raw.printSchema()

n_ratings_raw = ratings_raw.count()
n_movies_raw  = movies_raw.count()
print(f"Raw ratings rows : {n_ratings_raw:,}")
print(f"Raw movies rows  : {n_movies_raw:,}")
ratings_raw.show(5)
movies_raw.show(5, truncate=False)

# --------------------------------------------------------------------------
# 2. CLEAN & PREPROCESS
# --------------------------------------------------------------------------
banner("2. DATA CLEANING & PREPROCESSING")

print("Null counts - ratings:");  null_counts(ratings_raw).show()
print("Null counts - movies:");   null_counts(movies_raw).show()

dup_ratings = n_ratings_raw - ratings_raw.dropDuplicates(["userId", "movieId"]).count()
dup_movies  = n_movies_raw  - movies_raw.dropDuplicates(["movieId"]).count()
print(f"Duplicate (userId, movieId) ratings : {dup_ratings:,}")
print(f"Duplicate movieIds in movies        : {dup_movies:,}")

# 2.3 Clean movies
movies = (
    movies_raw
    .dropna(subset=["movieId", "title"])
    .dropDuplicates(["movieId"])
    .withColumn("title",      F.trim("title"))
    .withColumn("year_str",   F.regexp_extract("title", r"\((\d{4})\)\s*$", 1))
    .withColumn("year",       F.when(F.col("year_str") == "", F.lit(None))
                               .otherwise(F.col("year_str").cast("int")))
    .withColumn("year",       F.when(F.col("year").between(1870, 2025), F.col("year")))
    .withColumn("clean_title",F.trim(F.regexp_replace("title", r"\s*\(\d{4}\)\s*$", "")))
    .withColumn("genres",
        F.when(F.col("genres").isNull() | (F.col("genres") == "(no genres listed)"),
               F.lit("Unknown"))
         .otherwise(F.col("genres")))
    .withColumn("genre_list", F.split("genres", r"\|"))
    .withColumn("decade",     (F.floor(F.col("year") / 10) * 10).cast("int"))
    .drop("year_str")
)

# 2.4 Clean ratings
ratings = (
    ratings_raw
    .dropna(subset=["userId", "movieId", "rating", "timestamp"])
    .filter(F.col("rating").between(0.5, 5.0))
    .dropDuplicates(["userId", "movieId"])
    .join(F.broadcast(movies.select("movieId")), "movieId", "left_semi")
    .withColumn("rating_time", F.from_unixtime("timestamp").cast("timestamp"))
    .withColumn("rating_year", F.year("rating_time"))
    .select("userId", "movieId", "rating", "timestamp", "rating_time", "rating_year")
    .persist(StorageLevel.MEMORY_AND_DISK)
)

n_ratings = ratings.count()
n_movies  = movies.count()
print(f"Clean ratings rows : {n_ratings:,}  (removed {n_ratings_raw - n_ratings:,})")
print(f"Clean movies rows  : {n_movies:,}  (removed {n_movies_raw - n_movies:,})")
movies.show(5, truncate=False)

# --------------------------------------------------------------------------
# 3. ANALYSIS WITH SPARK DATAFRAME OPERATIONS
# --------------------------------------------------------------------------
banner("3. ANALYSIS - SPARK DATAFRAME OPERATIONS")

n_users       = ratings.select("userId").distinct().count()
n_rated_movies= ratings.select("movieId").distinct().count()
global_mean   = ratings.agg(F.avg("rating")).first()[0]
sparsity      = 1 - n_ratings / (n_users * n_rated_movies)
print(f"Users: {n_users:,} | Rated movies: {n_rated_movies:,} | Ratings: {n_ratings:,}")
print(f"Global average rating: {global_mean:.4f}")
print(f"Rating matrix sparsity: {sparsity:.4%}")
ratings.select("rating").describe().show()

rating_dist      = ratings.groupBy("rating").count().orderBy("rating")
rating_dist.show(10)

ratings_per_year = ratings.groupBy("rating_year").count().orderBy("rating_year")

# Per-movie statistics + weighted rating
movie_agg = ratings.groupBy("movieId").agg(
    F.count("*").alias("rating_count"),
    F.round(F.avg("rating"),    3).alias("avg_rating"),
    F.round(F.stddev("rating"), 3).alias("rating_stddev"),
)
movie_stats = movies.join(movie_agg, "movieId", "left").fillna({"rating_count": 0})

m_threshold = movie_stats.filter("rating_count > 0") \
                          .approxQuantile("rating_count", [0.90], 0.01)[0]
print(f"Weighted-rating threshold m (90th pct of rating_count): {m_threshold:.0f}")
v, R = F.col("rating_count"), F.coalesce(F.col("avg_rating"), F.lit(global_mean))
C, m = F.lit(global_mean), F.lit(m_threshold)
movie_stats = movie_stats.withColumn(
    "weighted_rating", F.round((v / (v + m)) * R + (m / (v + m)) * C, 4)
)
movie_stats = movie_stats.persist(StorageLevel.MEMORY_AND_DISK)

print("Top 10 most-rated movies:")
movie_stats.orderBy(F.desc("rating_count")) \
           .select("clean_title", "year", "rating_count", "avg_rating") \
           .show(10, truncate=False)

print("Top 10 highest weighted-rated movies:")
movie_stats.orderBy(F.desc("weighted_rating")) \
           .select("clean_title", "year", "rating_count", "avg_rating", "weighted_rating") \
           .show(10, truncate=False)

# Genre analysis
movie_genres = movies.select("movieId", "clean_title", F.explode("genre_list").alias("genre"))
genre_stats = (
    movie_genres
    .join(movie_stats.select("movieId", "rating_count", "avg_rating"), "movieId")
    .groupBy("genre")
    .agg(
        F.countDistinct("movieId").alias("num_movies"),
        F.sum("rating_count").alias("total_ratings"),
        F.round(
            F.sum(F.col("avg_rating") * F.col("rating_count")) / F.sum("rating_count"), 3
        ).alias("avg_rating"),
    )
    .orderBy(F.desc("total_ratings"))
)
print("Genre statistics:")
genre_stats.show(25, truncate=False)

# User activity
user_stats = ratings.groupBy("userId").agg(
    F.count("*").alias("num_ratings"),
    F.round(F.avg("rating"), 3).alias("avg_rating"),
)
user_stats.describe("num_ratings", "avg_rating").show()

# Release decade
decade_stats = (
    movie_stats.filter(F.col("decade").isNotNull() & (F.col("rating_count") > 0))
    .groupBy("decade")
    .agg(
        F.count("*").alias("num_movies"),
        F.sum("rating_count").alias("total_ratings"),
        F.round(
            F.sum(F.col("avg_rating") * F.col("rating_count")) / F.sum("rating_count"), 3
        ).alias("avg_rating"),
    )
    .orderBy("decade")
)

# --------------------------------------------------------------------------
# 4. SPARK SQL
# --------------------------------------------------------------------------
banner("4. SPARK SQL")

ratings.createOrReplaceTempView("ratings")
movies.createOrReplaceTempView("movies")
movie_stats.createOrReplaceTempView("movie_stats")
movie_genres.createOrReplaceTempView("movie_genres")
user_stats.createOrReplaceTempView("user_stats")

print("SQL 1: Top 10 most active users")
spark.sql("""
    SELECT userId, num_ratings, avg_rating
    FROM user_stats ORDER BY num_ratings DESC LIMIT 10
""").show()

print("SQL 2: Avg rating and num ratings per genre (min 1 000 ratings)")
spark.sql("""
    SELECT g.genre,
           COUNT(DISTINCT g.movieId) AS num_movies,
           SUM(s.rating_count)       AS total_ratings,
           ROUND(SUM(s.avg_rating * s.rating_count) / SUM(s.rating_count), 3) AS avg_rating
    FROM movie_genres g JOIN movie_stats s ON g.movieId = s.movieId
    GROUP BY g.genre HAVING SUM(s.rating_count) >= 1000
    ORDER BY avg_rating DESC
""").show(25, truncate=False)

print("SQL 3: Top 3 movies in each genre (min 500 ratings, ranked by weighted rating)")
spark.sql("""
    SELECT genre, clean_title, year, rating_count, avg_rating, weighted_rating, rnk
    FROM (
        SELECT g.genre, s.clean_title, s.year, s.rating_count, s.avg_rating,
               s.weighted_rating,
               ROW_NUMBER() OVER (PARTITION BY g.genre ORDER BY s.weighted_rating DESC) AS rnk
        FROM movie_genres g JOIN movie_stats s ON g.movieId = s.movieId
        WHERE s.rating_count >= 500
    ) t WHERE rnk <= 3
    ORDER BY genre, rnk
""").show(60, truncate=False)

print("SQL 4: Ratings given per year")
spark.sql("""
    SELECT rating_year, COUNT(*) AS num_ratings, ROUND(AVG(rating), 3) AS avg_rating
    FROM ratings GROUP BY rating_year ORDER BY rating_year
""").show(40)

print("SQL 5: Movies per release decade with average rating")
spark.sql("""
    SELECT decade, COUNT(*) AS num_movies,
           ROUND(SUM(avg_rating * rating_count) / SUM(rating_count), 3) AS avg_rating
    FROM movie_stats WHERE decade IS NOT NULL AND rating_count > 0
    GROUP BY decade ORDER BY decade
""").show(20)

print("SQL 6: Users who rated at least 1 000 movies")
spark.sql("SELECT COUNT(*) AS heavy_users FROM user_stats WHERE num_ratings >= 1000").show()

# --------------------------------------------------------------------------
# 5. VISUALISATIONS (Matplotlib)
# --------------------------------------------------------------------------
banner("5. VISUALISATIONS")

# 5.1 Rating distribution
pdf = rating_dist.toPandas()
plt.figure(figsize=(8, 5))
plt.bar(pdf["rating"].astype(str), pdf["count"] / 1e6, color="#4C72B0")
plt.title("Rating Distribution (MovieLens 32M)")
plt.xlabel("Rating"); plt.ylabel("Number of ratings (millions)")
save_fig("01_rating_distribution.png")

# 5.2 Ratings per year
pdf = ratings_per_year.toPandas()
plt.figure(figsize=(10, 5))
plt.plot(pdf["rating_year"], pdf["count"] / 1e6, marker="o", color="#DD8452")
plt.title("Number of Ratings per Year")
plt.xlabel("Year"); plt.ylabel("Ratings (millions)"); plt.grid(alpha=0.3)
save_fig("02_ratings_per_year.png")

# 5.3 Genre popularity & 5.4 avg rating by genre
gpdf = genre_stats.toPandas()
gpdf = gpdf[gpdf["genre"] != "Unknown"]

top = gpdf.sort_values("total_ratings")
plt.figure(figsize=(9, 7))
plt.barh(top["genre"], top["total_ratings"] / 1e6, color="#55A868")
plt.title("Genre Popularity (total ratings)"); plt.xlabel("Ratings (millions)")
save_fig("03_genre_popularity.png")

top = gpdf.sort_values("avg_rating")
plt.figure(figsize=(9, 7))
plt.barh(top["genre"], top["avg_rating"], color="#C44E52")
plt.xlim(top["avg_rating"].min() - 0.2, top["avg_rating"].max() + 0.1)
plt.title("Average Rating by Genre"); plt.xlabel("Average rating")
save_fig("04_avg_rating_by_genre.png")

# 5.5 Top 15 movies by weighted rating
tpdf = (
    movie_stats.orderBy(F.desc("weighted_rating"))
    .select("clean_title", "year", "weighted_rating").limit(15).toPandas()
    .sort_values("weighted_rating")
)
labels = tpdf["clean_title"].str.slice(0, 40) + " (" + tpdf["year"].astype("Int64").astype(str) + ")"
plt.figure(figsize=(10, 7))
plt.barh(labels, tpdf["weighted_rating"], color="#8172B3")
plt.xlim(tpdf["weighted_rating"].min() - 0.1, 5)
plt.title("Top 15 Movies by Weighted Rating"); plt.xlabel("Weighted rating")
save_fig("05_top_movies_weighted.png")

# 5.6 Ratings per user (log scale)
upd = user_stats.select("num_ratings").toPandas()
plt.figure(figsize=(8, 5))
plt.hist(upd["num_ratings"], bins=60, color="#64B5CD", log=True)
plt.title("Ratings per User (log scale)")
plt.xlabel("Number of ratings by a user"); plt.ylabel("Number of users")
save_fig("06_ratings_per_user.png")

# 5.7 Movies per decade
dpdf = decade_stats.toPandas()
plt.figure(figsize=(9, 5))
plt.bar(dpdf["decade"].astype(str) + "s", dpdf["num_movies"], color="#937860")
plt.title("Movies by Release Decade")
plt.xlabel("Decade"); plt.ylabel("Number of movies"); plt.xticks(rotation=45)
save_fig("07_movies_per_decade.png")

# 5.8 Popularity vs average rating (sampled)
spdf = (movie_stats.filter("rating_count >= 20")
        .select("rating_count", "avg_rating").sample(0.3, seed=42).toPandas())
plt.figure(figsize=(8, 5))
plt.scatter(spdf["rating_count"], spdf["avg_rating"], s=4, alpha=0.3, color="#4C72B0")
plt.xscale("log")
plt.title("Movie Popularity vs Average Rating")
plt.xlabel("Number of ratings (log scale)"); plt.ylabel("Average rating")
save_fig("08_popularity_vs_rating.png")

# --------------------------------------------------------------------------
# 6. EXPORT CLEANED DATA (Windows-safe via local temp → copy)
# --------------------------------------------------------------------------
banner("6. EXPORTING CLEANED DATASETS")

export_cols = ["movieId", "title", "clean_title", "year", "decade", "genres",
               "rating_count", "avg_rating", "weighted_rating"]

# 6.1 ratings_clean.parquet  (ALS needs userId, movieId, rating, timestamp as ints/floats)
print("\n[1/4] Writing ratings_clean.parquet ...")
safe_parquet_write(
    ratings.select("userId", "movieId", "rating", "timestamp"),
    os.path.join(CLEAN_DIR, "ratings_clean.parquet"),
    n_partitions=1,
)

# 6.2 movies_clean.parquet
print("\n[2/4] Writing movies_clean.parquet ...")
safe_parquet_write(
    movie_stats.select(*export_cols),
    os.path.join(CLEAN_DIR, "movies_clean.parquet"),
    n_partitions=1,
)

# 6.3 movies_clean.csv  (Streamlit / pandas friendly)
print("\n[3/4] Writing movies_clean.csv ...")
movie_csv = os.path.join(CLEAN_DIR, "movies_clean.csv")
movie_stats.select(*export_cols).orderBy(F.desc("weighted_rating")).toPandas().to_csv(
    movie_csv, index=False
)
print(f"  saved -> {movie_csv}")

# 6.4 genre_stats.csv
print("\n[4/4] Writing genre_stats.csv ...")
genre_csv = os.path.join(CLEAN_DIR, "genre_stats.csv")
genre_stats.toPandas().to_csv(genre_csv, index=False)
print(f"  saved -> {genre_csv}")

# ── Final summary ─────────────────────────────────────────────────────────
banner("OUTPUT SUMMARY")
for fname in ["ratings_clean.parquet", "movies_clean.parquet",
              "movies_clean.csv", "genre_stats.csv"]:
    full = os.path.join(CLEAN_DIR, fname)
    if os.path.isdir(full):
        size = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, files in os.walk(full) for f in files
        )
        print(f"  ✓ {fname:35s}  {size/1e6:>8.2f} MB  (folder)")
    elif os.path.isfile(full):
        print(f"  ✓ {fname:35s}  {os.path.getsize(full)/1e6:>8.2f} MB")
    else:
        print(f"  ✗ {fname:35s}  MISSING")

banner("DONE - Module 1 complete")
spark.stop()

# Clean up Spark tmp
try:
    shutil.rmtree(_SPARK_TMP, ignore_errors=True)
except Exception:
    pass
