"""
Speed Layer — Spark Structured Streaming (TAM VERSİYON).

Yapılanlar:
  1. Kafka'dan air_quality_events topic'ini stream olarak oku
  2. JSON parse + şema uygulama
  3. Watermark (10 dk) + dedup (event_id üzerinden)
  4. Tumbling window (5 dk) ile istasyon × parametre ortalamaları
  5. Kuralcı anomali flag (PM2.5 > 150, CO > 25, NO2 > 200, ...)
  6. 3 paralel sink:
       - console (sadece anomali alarmları, debug için)
       - parquet/raw      (batch layer kaynağı)
       - parquet/aggregates (5dk pencere ortalamaları)

Çalıştırma:
  spark-submit \
      --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
      speed_layer/streaming_app.py
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, to_timestamp, window, avg, count,
    when, lit, expr,
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, TimestampType,
)

# --- Yapılandırma ---
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC_AIR", "air_quality_events")

DATA_DIR = "data/raw_parquet"
RAW_PATH = f"{DATA_DIR}/events"
AGG_PATH = f"{DATA_DIR}/aggregates"
CKPT_RAW = "data/checkpoints/raw_events"
CKPT_AGG = "data/checkpoints/aggregates"
CKPT_CONSOLE = "data/checkpoints/console_alerts"

# Kuralcı anomali eşikleri (AQICN'in iaqi cinsinden — birim parametreye göre değişir,
# genel pratik: sağlıksız AQI band'leri civarı)
RULE_THRESHOLDS = {
    "pm25": 150.0,
    "pm10": 200.0,
    "co":   25.0,
    "no2":  200.0,
    "so2":  150.0,
    "o3":   150.0,
}

# --- Şema ---
event_schema = StructType([
    StructField("source",        StringType(), True),
    StructField("city",          StringType(), True),
    StructField("station_id",    StringType(), True),
    StructField("station_name",  StringType(), True),
    StructField("lat",           DoubleType(), True),
    StructField("lon",           DoubleType(), True),
    StructField("measured_at",   StringType(), True),  # ISO string, parse edilecek
    StructField("ingested_at",   StringType(), True),
    StructField("aqi",           DoubleType(), True),
    StructField("parameter",     StringType(), True),
    StructField("value",         DoubleType(), True),
    StructField("event_id",      StringType(), True),
])


def build_spark():
    return (
        SparkSession.builder
        .appName("AirQualitySpeedLayer")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    print(f"Spark: {spark.version} | Kafka: {KAFKA_BOOTSTRAP} | Topic: {TOPIC}")

    # 1) Kafka'dan oku
    raw_df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .load()
    )

    # 2) JSON parse + tipli kolonlar
    parsed_df = (
        raw_df
        .select(
            col("key").cast("string").alias("kafka_key"),
            from_json(col("value").cast("string"), event_schema).alias("ev"),
            col("timestamp").alias("kafka_ts"),
        )
        .select("kafka_key", "kafka_ts", "ev.*")
        # measured_at string ISO formatında geliyor — Timestamp'a çevir
        .withColumn("event_time", to_timestamp(col("measured_at")))
        .withColumn("ingested_time", to_timestamp(col("ingested_at")))
    )

    # 3) Watermark + dedup
    # event_id ile dedup, watermark 10 dakikalık geç gelme toleransı
    deduped_df = (
        parsed_df
        .withWatermark("event_time", "10 minutes")
        .dropDuplicates(["event_id"])
    )

    # 4) Anomali bayrağı (kuralcı, ileride IF UDF ile değiştirilecek)
    threshold_expr = " ".join([
        f"WHEN parameter = '{p}' AND value > {t} THEN true"
        for p, t in RULE_THRESHOLDS.items()
    ])
    flagged_df = deduped_df.withColumn(
        "is_outlier",
        expr(f"CASE {threshold_expr} ELSE false END")
    )

    # 5a) RAW SINK — tüm dedup'lı event'ler Parquet'e (batch layer kaynağı)
    raw_query = (
        flagged_df.writeStream
        .format("parquet")
        .option("path", RAW_PATH)
        .option("checkpointLocation", CKPT_RAW)
        .partitionBy("city", "parameter")
        .outputMode("append")
        .trigger(processingTime="30 seconds")
        .queryName("raw_events_sink")
        .start()
    )

    # 5b) WINDOWED AGGREGATES — 5 dakikalık tumbling window, istasyon × parametre ortalaması
    agg_df = (
        deduped_df
        .groupBy(
            window(col("event_time"), "5 minutes"),
            col("city"),
            col("station_id"),
            col("station_name"),
            col("parameter"),
        )
        .agg(
            avg("value").alias("value_avg"),
            count("*").alias("event_count"),
        )
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            "city", "station_id", "station_name", "parameter",
            "value_avg", "event_count",
        )
    )

    agg_query = (
        agg_df.writeStream
        .format("parquet")
        .option("path", AGG_PATH)
        .option("checkpointLocation", CKPT_AGG)
        .partitionBy("city", "parameter")
        .outputMode("append")
        .trigger(processingTime="30 seconds")
        .queryName("aggregates_sink")
        .start()
    )

    # 5c) CONSOLE SINK — sadece anomali alarmları
    alerts_df = (
        flagged_df
        .filter(col("is_outlier") == True)
        .select(
            "event_time", "city", "station_name", "parameter", "value",
            "aqi", "is_outlier",
        )
    )
    alerts_query = (
        alerts_df.writeStream
        .format("console")
        .option("truncate", "false")
        .option("numRows", 20)
        .option("checkpointLocation", CKPT_CONSOLE)
        .outputMode("append")
        .trigger(processingTime="30 seconds")
        .queryName("alerts_console")
        .start()
    )

    print("\n=== 3 streaming query başladı ===")
    print(f"  Raw events  → {RAW_PATH}")
    print(f"  Aggregates  → {AGG_PATH}")
    print(f"  Alerts      → console (sadece is_outlier=true)\n")
    print("Ctrl+C ile durdur.\n")

    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()