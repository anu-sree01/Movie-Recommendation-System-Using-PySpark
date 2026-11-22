"""
Member 2 - SparkSession Management
Module: member2_recommendation/src/spark_session.py
"""

import os
import sys
import tempfile
import shutil
from pyspark.sql import SparkSession


def get_spark_session(
    app_name: str = "MovieLens32M-Member2-ALS",
    master: str = "local[*]",
    driver_memory: str = "3g",
    shuffle_partitions: int = 32,
    temp_dir_prefix: str = "spark_member2_als_",
) -> tuple[SparkSession, str]:
    """
    Initializes a Windows-safe, resource-aware SparkSession.
    
    Args:
        app_name: Descriptive name for the Spark application.
        master: Spark master URL (default 'local[*]').
        driver_memory: Driver memory limit (default '3g', safe for 8GB total RAM).
        shuffle_partitions: Number of partitions for SQL/DataFrame shuffles.
        temp_dir_prefix: Prefix for local staging temporary directory.
        
    Returns:
        tuple of (SparkSession, temp_dir_path)
    """
    # 1. Point PySpark workers explicitly to active Python interpreter
    _python_exe = sys.executable
    os.environ["PYSPARK_PYTHON"] = _python_exe
    os.environ["PYSPARK_DRIVER_PYTHON"] = _python_exe

    # 2. Configure HADOOP_HOME dynamically for Windows compatibility
    _local_hadoop = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".hadoop"))
    if os.path.isdir(_local_hadoop):
        _hadoop_home = _local_hadoop
    else:
        _hadoop_home = os.environ.get("HADOOP_HOME", os.path.expanduser("~/.hadoop"))

    if os.path.isdir(_hadoop_home):
        os.environ["HADOOP_HOME"] = _hadoop_home
        _bin = os.path.join(_hadoop_home, "bin")
        if _bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")
        print(f"[spark_session] Configured HADOOP_HOME = {_hadoop_home}")

    # 3. Allocate isolated temporary directory for local Windows staging
    temp_dir = tempfile.mkdtemp(prefix=temp_dir_prefix)
    warehouse_dir = os.path.join(temp_dir, "warehouse")
    checkpoint_dir = os.path.join(temp_dir, "checkpoints")
    os.makedirs(warehouse_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 3. Build SparkSession with defensive memory and filesystem settings
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.local.dir", temp_dir)
        .config("spark.sql.warehouse.dir", warehouse_dir)
        .config("spark.driver.extraJavaOptions",
                f"-Djava.io.tmpdir={temp_dir} -Dderby.system.home={temp_dir}")
        .config("spark.speculation", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    return spark, temp_dir


def stop_spark_session(spark: SparkSession, temp_dir: str = None) -> None:
    """Stops the SparkSession and safely purges temporary directories."""
    if spark:
        try:
            spark.stop()
        except Exception:
            pass

    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
