"""
Member 2 - Environment Verification and Spark Smoke Test
Module: member2_recommendation/src/smoke_test.py
"""

import os
import sys
import tempfile
import shutil

print("=" * 80)
print("MEMBER 2: ENVIRONMENT AND SPARK SMOKE TEST")
print("=" * 80)

# 1. Environment Details
print(f"Python Executable : {sys.executable}")
print(f"Python Version    : {sys.version.split()[0]}")

# 2. PySpark & Py4J Imports
try:
    import py4j
    print(f"Py4J Version      : {py4j.__version__}")
except Exception as e:
    print(f"ERROR importing Py4J: {e}")
    sys.exit(1)

try:
    import pyspark
    print(f"PySpark Version   : {pyspark.__version__}")
except Exception as e:
    print(f"ERROR importing PySpark: {e}")
    sys.exit(1)

# 3. Test ALS & MLlib imports
try:
    from pyspark.ml.recommendation import ALS
    from pyspark.ml.evaluation import RegressionEvaluator
    print("PySpark MLlib ALS Import : SUCCESS (ALS & RegressionEvaluator imported)")
except Exception as e:
    print(f"ERROR importing ALS from pyspark.ml.recommendation: {e}")
    sys.exit(1)

# 4. Windows / Temp Directory Setup
_PYTHON_EXE = sys.executable
os.environ["PYSPARK_PYTHON"] = _PYTHON_EXE
os.environ["PYSPARK_DRIVER_PYTHON"] = _PYTHON_EXE

_SPARK_TMP = tempfile.mkdtemp(prefix="spark_member2_smoke_")
print(f"Spark Temp Directory : {_SPARK_TMP}")

# 5. Initialize Minimal SparkSession
from pyspark.sql import SparkSession

# Driver memory set to 3g (safe for 8GB system RAM)
print("Initializing SparkSession (master=local[*], driver-memory=3g)...")
spark = (
    SparkSession.builder
    .appName("MovieLens32M-Member2-SmokeTest")
    .master("local[*]")
    .config("spark.driver.memory", "3g")
    .config("spark.sql.shuffle.partitions", "32")
    .config("spark.sql.session.timeZone", "UTC")
    .config("spark.local.dir", _SPARK_TMP)
    .config("spark.sql.warehouse.dir", os.path.join(_SPARK_TMP, "warehouse"))
    .config("spark.driver.extraJavaOptions",
            f"-Djava.io.tmpdir={_SPARK_TMP} -Dderby.system.home={_SPARK_TMP}")
    .config("spark.speculation", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print(f"SparkSession Status: SUCCESS (Spark version: {spark.version})")
jvm_version = spark._jvm.java.lang.System.getProperty("java.version")
print(f"JVM Java Version   : {jvm_version}")

# 6. Locate Clean Datasets (Resolves part file directly to prevent Windows NativeIO directory listing issue)
def resolve_parquet_target(candidates):
    for base in candidates:
        if os.path.isdir(base):
            parts = [f for f in os.listdir(base) if f.endswith(".parquet") and not f.startswith(".")]
            if parts:
                return os.path.join(base, parts[0])
            return base
        elif os.path.isfile(base):
            return base
    return None

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

ratings_path = resolve_parquet_target(candidates_ratings)
movies_path  = resolve_parquet_target(candidates_movies)

if not ratings_path or not movies_path:
    print(f"ERROR: Could not locate parquet files. Ratings: {ratings_path}, Movies: {movies_path}")
    spark.stop()
    sys.exit(1)

print(f"\nResolved Ratings Target: {ratings_path}")
print(f"Resolved Movies Target : {movies_path}")

# 7. Read Datasets & Verify Schemas and Partitions
print("\n--- Reading ratings_clean.parquet ---")
ratings_df = spark.read.parquet(ratings_path)
print("Ratings Schema:")
ratings_df.printSchema()
ratings_partitions = ratings_df.rdd.getNumPartitions()
print(f"Ratings Partition Count: {ratings_partitions}")
print("Ratings Sample (5 rows without full collect):")
ratings_df.show(5)

print("\n--- Reading movies_clean.parquet ---")
movies_df = spark.read.parquet(movies_path)
print("Movies Schema:")
movies_df.printSchema()
movies_partitions = movies_df.rdd.getNumPartitions()
print(f"Movies Partition Count: {movies_partitions}")
print("Movies Sample (5 rows without full collect):")
movies_df.show(5, truncate=False)

# 8. Record Count Verification via Spark action
print("\nVerifying Spark actions without Pandas collect:")
n_r = ratings_df.count()
n_m = movies_df.count()
print(f"Verified Ratings Count : {n_r:,}")
print(f"Verified Movies Count  : {n_m:,}")

print("\n" + "=" * 80)
print("SMOKE TEST COMPLETED SUCCESSFULLY!")
print("=" * 80)

spark.stop()

try:
    shutil.rmtree(_SPARK_TMP, ignore_errors=True)
except Exception:
    pass
