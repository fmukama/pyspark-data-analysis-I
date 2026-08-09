import os

from pyspark.sql import SparkSession

from src import config
from src.logger import get_logger

logger = get_logger("spark_session")

_WAREHOUSE_DIR = os.path.join(config.BASE_DIR, "data", "spark-warehouse")


def get_spark_session() -> SparkSession:
    """Return the project's SparkSession, building it on first call."""
    spark = (
        SparkSession.builder
        .appName(config.SPARK_APP_NAME)
        .master(config.SPARK_MASTER)
        .config("spark.sql.shuffle.partitions", config.SPARK_SHUFFLE_PARTITIONS)
        .config("spark.sql.warehouse.dir", _WAREHOUSE_DIR)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info(
        "SparkSession ready: app=%s master=%s shuffle.partitions=%s warehouse=%s",
        config.SPARK_APP_NAME, config.SPARK_MASTER, config.SPARK_SHUFFLE_PARTITIONS, _WAREHOUSE_DIR,
    )
    return spark
