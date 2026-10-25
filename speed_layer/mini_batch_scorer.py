"""
Mini-Batch Scorer — Lambda mimarisinin "scoring" katmanı.

Her N dakikada bir:
  1. Streaming layer'ın yazdığı raw Parquet event'leri okur
  2. Long → wide pivot ile istasyon × saat × parametre tablosu kurar
  3. Eğitilmiş Isolation Forest model + scaler ile her satıra anomali skoru atar
  4. Skorlanmış sonucu data/raw_parquet/scored/ altına yazar
  5. Anomali olanları (-1 prediction) anomaly_alerts Kafka topic'ine yollar

Stateful: hangi event_time'a kadar işlendiğini /data/checkpoints/scorer_state.json'da tutar.
Bu sayede iki tur arasında sadece YENİ event'ler işlenir, idempotent çalışır.

Çalıştırma:
  python -m speed_layer.mini_batch_scorer
"""
import os
import json
import time
import signal
import logging
# TensorFlow log spam'ini bastır
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from kafka import KafkaProducer
from kafka.errors import KafkaError

from batch_layer.data_prep import (
    long_to_wide,
    add_engineered_features,
    filter_complete_rows,
    impute_missing,
    ALL_FEATURES,
    CORE_FEATURES,
)

# --- Yapılandırma ---
load_dotenv()
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
ALERTS_TOPIC = "anomaly_alerts"

RAW_EVENTS_PATH = "data/raw_parquet/events"
SCORED_PATH = "data/raw_parquet/scored"
STATE_PATH = Path("data/checkpoints/scorer_state.json")

IF_MODEL_PATH = Path("data/models/isolation_forest.joblib")
IF_SCALER_PATH = Path("data/models/if_scaler.joblib")
IF_META_PATH = Path("data/models/if_metadata.json")

# LSTM AE — Versiyon B (FULL, 13 şehir)
LSTM_MODEL_PATH = Path("data/models/lstm_autoencoder_full.keras")
LSTM_SCALER_PATH = Path("data/models/lstm_scaler_full.joblib")
LSTM_META_PATH = Path("data/models/lstm_metadata_full.json")
LSTM_SEQUENCE_LEN = 12  # Eğitim sırasında kullanılan değer

POLL_INTERVAL_SEC = int(os.getenv("SCORER_POLL_INTERVAL_SEC", "300"))
LOOKBACK_HOURS = int(os.getenv("SCORER_LOOKBACK_HOURS", "6"))

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scorer")

# Graceful shutdown
_shutdown = False
def _handle_sigint(signum, frame):
    global _shutdown
    log.info("Shutdown sinyali alındı, mevcut tur bitince çıkacak.")
    _shutdown = True
signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)


# --- Yardımcılar ---
def load_state():
    if not STATE_PATH.exists():
        return {"last_processed_event_time": None, "last_run_at": None}
    with open(STATE_PATH, "r") as f:
        return json.load(f)


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def load_models():
    """IF + LSTM AE modellerini, scaler'ları ve metadata'larını yükle."""
    # --- Isolation Forest ---
    if not IF_MODEL_PATH.exists():
        raise RuntimeError(f"IF model bulunamadı: {IF_MODEL_PATH}. Önce train_isolation_forest çalıştır.")
    log.info(f"IF model yükleniyor: {IF_MODEL_PATH}")
    if_model = joblib.load(IF_MODEL_PATH)
    if_scaler = joblib.load(IF_SCALER_PATH)
    with open(IF_META_PATH) as f:
        if_meta = json.load(f)
    log.info(f"  IF | Eğitim: {if_meta['trained_at']}, train sample: {if_meta['n_train_samples']}")

    # --- LSTM Autoencoder (opsiyonel — yoksa sadece IF kullanılır) ---
    lstm_model = None
    lstm_scaler = None
    lstm_meta = None
    if LSTM_MODEL_PATH.exists():
        log.info(f"LSTM AE model yükleniyor: {LSTM_MODEL_PATH}")
        lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH, compile=False)
        lstm_scaler = joblib.load(LSTM_SCALER_PATH)
        with open(LSTM_META_PATH) as f:
            lstm_meta = json.load(f)
        log.info(
            f"  LSTM | Eğitim: {lstm_meta['trained_at']}, "
            f"sequences: {lstm_meta['n_train_sequences']}, "
            f"threshold p99: {lstm_meta['threshold_p99']:.4f}"
        )
    else:
        log.warning(f"LSTM AE model bulunamadı ({LSTM_MODEL_PATH}). Sadece IF kullanılacak.")

    return {
        "if_model": if_model,
        "if_scaler": if_scaler,
        "if_meta": if_meta,
        "lstm_model": lstm_model,
        "lstm_scaler": lstm_scaler,
        "lstm_meta": lstm_meta,
    }


def make_kafka_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks="all",
        retries=5,
        linger_ms=50,
    )


def score_recent_events(models, since_event_time):
    """
    Son N saatlik event'leri al, wide'a çevir, IF + LSTM AE ile skorla.
    Returns scored_df (wide format + if_score, lstm_mse, is_anomaly_if, is_anomaly_lstm
    sütunları) veya None.
    """
    if not Path(RAW_EVENTS_PATH).exists():
        log.warning("Raw events path yok, atlanıyor.")
        return None

    df_long = pd.read_parquet(RAW_EVENTS_PATH)
    df_long["event_time"] = pd.to_datetime(df_long["event_time"], utc=True)

    # Lookback filtresi
    now = datetime.now(timezone.utc)
    cutoff = now - pd.Timedelta(hours=LOOKBACK_HOURS)
    df_long = df_long[df_long["event_time"] >= cutoff]

    if df_long.empty:
        log.info("Son periyotta event yok.")
        return None

    # Long → wide
    wide = long_to_wide(df_long)
    if wide.empty:
        log.info("Wide formata çevrilebilir satır kalmadı.")
        return None

    wide = add_engineered_features(wide)
    wide = filter_complete_rows(wide, min_features=8)
    if wide.empty:
        log.info("Filter sonrası boş.")
        return None
    wide = impute_missing(wide)

    # YENİ satırlar
    if since_event_time is not None:
        since_dt = pd.to_datetime(since_event_time, utc=True)
        wide = wide[wide["event_time"] > since_dt]
        if wide.empty:
            log.info(f"Yeni satır yok (state cutoff: {since_event_time}).")
            return None

    # ============================================================
    # 1) ISOLATION FOREST — her satıra direkt skor
    # ============================================================
    if_meta = models["if_meta"]
    if_feature_cols = if_meta["feature_columns"]
    X_if = models["if_scaler"].transform(wide[if_feature_cols].values)
    if_pred = models["if_model"].predict(X_if)              # +1 / -1
    if_score = models["if_model"].score_samples(X_if)       # büyük → daha normal

    scored = wide.copy()
    scored["if_score"] = if_score
    scored["if_pred"] = if_pred
    scored["is_anomaly_if"] = (if_pred == -1)

    # ============================================================
    # 2) LSTM AE — istasyon başına sequence kur, reconstruction MSE
    # ============================================================
    if models["lstm_model"] is not None:
        lstm_meta = models["lstm_meta"]
        lstm_feature_cols = lstm_meta["feature_columns"]
        lstm_threshold = lstm_meta["threshold_p99"]

        # Tüm Parquet'ten sequence çıkarmak gerek (lookback dışındakiler de
        # son 12 saatlik penceredeki bir noktayı tamamlamak için lazım olabilir)
        # Bu yüzden full Parquet'i oku, son 24 saat al
        df_full = pd.read_parquet(RAW_EVENTS_PATH)
        df_full["event_time"] = pd.to_datetime(df_full["event_time"], utc=True)
        full_cutoff = now - pd.Timedelta(hours=LOOKBACK_HOURS + LSTM_SEQUENCE_LEN)
        df_full = df_full[df_full["event_time"] >= full_cutoff]
        wide_full = long_to_wide(df_full)
        if not wide_full.empty:
            wide_full = add_engineered_features(wide_full)
            wide_full = filter_complete_rows(wide_full, min_features=8)
            wide_full = impute_missing(wide_full)

            # Scaling
            wide_scaled = wide_full.copy()
            wide_scaled[lstm_feature_cols] = models["lstm_scaler"].transform(
                wide_full[lstm_feature_cols].values
            )

            # İstasyon başına sequence çıkar (sadece skor isteyen satırlar için)
            lstm_scores = {}  # (station_id, end_time) -> mse
            for station_id, grp in wide_scaled.sort_values(
                ["station_id", "event_time"]
            ).groupby("station_id", observed=True):
                values = grp[lstm_feature_cols].values
                times = grp["event_time"].values
                if len(values) < LSTM_SEQUENCE_LEN:
                    continue
                # Her sequence: [t-11..t]
                seqs = []
                ends = []
                for i in range(len(values) - LSTM_SEQUENCE_LEN + 1):
                    seqs.append(values[i : i + LSTM_SEQUENCE_LEN])
                    ends.append(times[i + LSTM_SEQUENCE_LEN - 1])
                seqs = np.array(seqs)

                # Reconstruction
                recon = models["lstm_model"].predict(seqs, verbose=0)
                mses = np.mean(np.square(seqs - recon), axis=(1, 2))

                for end_time, mse in zip(ends, mses):
                    # Timezone-aware nanosecond timestamp olarak kaydet
                    key_ts = pd.Timestamp(end_time).tz_convert("UTC") if pd.Timestamp(end_time).tzinfo else pd.Timestamp(end_time, tz="UTC")
                    lstm_scores[(station_id, key_ts)] = float(mse)

            def _lookup_lstm(row):
                ts = pd.Timestamp(row["event_time"])
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                return lstm_scores.get((row["station_id"], ts), np.nan)

            scored["lstm_mse"] = scored.apply(_lookup_lstm, axis=1)
            scored["is_anomaly_lstm"] = scored["lstm_mse"].fillna(0.0) > lstm_threshold
        else:
            scored["lstm_mse"] = np.nan
            scored["is_anomaly_lstm"] = False
    else:
        scored["lstm_mse"] = np.nan
        scored["is_anomaly_lstm"] = False

    # Birleşik anomali: ya IF ya da LSTM herhangi biri işaretlerse
    scored["is_anomaly"] = scored["is_anomaly_if"] | scored["is_anomaly_lstm"].fillna(False)
    scored["scored_at"] = now.isoformat()

    return scored


def write_scored_parquet(scored: pd.DataFrame):
    """Skorlanmış sonuçları partition'lı Parquet olarak yaz (append)."""
    out_path = Path(SCORED_PATH)
    out_path.mkdir(parents=True, exist_ok=True)

    # Her batch için ayrı dosya — partition by city
    timestamp_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    # observed=True → categorical kolonda boş gruplar iter edilmesin
    for city, grp in scored.groupby("city", observed=True):
        if grp.empty:
            continue
        city_dir = out_path / f"city={city}"
        city_dir.mkdir(parents=True, exist_ok=True)
        out_file = city_dir / f"part-{timestamp_tag}.parquet"
        grp.to_parquet(out_file, index=False)


def emit_alerts(producer, scored: pd.DataFrame):
    """Anomali olan satırları anomaly_alerts topic'ine yolla."""
    anomalies = scored[scored["is_anomaly"]]
    sent = 0
    for _, row in anomalies.iterrows():
        alert = {
            "event_time": row["event_time"].isoformat() if pd.notna(row["event_time"]) else None,
            "city": row["city"],
            "station_id": row["station_id"],
            "station_name": row["station_name"],
            "lat": float(row["lat"]) if pd.notna(row["lat"]) else None,
            "lon": float(row["lon"]) if pd.notna(row["lon"]) else None,
            "if_score": float(row["if_score"]),
            "is_anomaly_if": bool(row["is_anomaly_if"]),
            "lstm_mse": float(row["lstm_mse"]) if pd.notna(row["lstm_mse"]) else None,
            "is_anomaly_lstm": bool(row["is_anomaly_lstm"]) if pd.notna(row["is_anomaly_lstm"]) else False,
            "is_anomaly": True,
            "feature_snapshot": {f: float(row[f]) for f in CORE_FEATURES if pd.notna(row[f])},
            "scored_at": row["scored_at"],
        }
        try:
            producer.send(ALERTS_TOPIC, key=str(row["station_id"]), value=alert)
            sent += 1
        except KafkaError as e:
            log.error(f"Kafka send hata: {e}")
    if sent:
        producer.flush()
    return sent, len(anomalies)


def main():
    log.info("Mini-Batch Scorer başlıyor")
    models = load_models()
    producer = make_kafka_producer()
    log.info(f"Kafka producer hazır → {KAFKA_BOOTSTRAP} / topic={ALERTS_TOPIC}")
    log.info(f"Polling interval: {POLL_INTERVAL_SEC}s, lookback: {LOOKBACK_HOURS}h")

    state = load_state()
    log.info(f"State yüklendi: last_processed_event_time={state.get('last_processed_event_time')}")

    round_num = 0
    while not _shutdown:
        round_num += 1
        log.info(f"=== Tur #{round_num} başlıyor ===")

        try:
            scored = score_recent_events(
                models,
                since_event_time=state.get("last_processed_event_time"),
            )
        except Exception as e:
            log.exception(f"Scoring hata: {e}")
            scored = None

        if scored is not None and not scored.empty:
            n_total = len(scored)
            n_anom_if = int(scored["is_anomaly_if"].sum())
            n_anom_lstm = int(scored["is_anomaly_lstm"].fillna(False).sum())
            n_anom_combined = int(scored["is_anomaly"].sum())

            write_scored_parquet(scored)
            sent, alert_count = emit_alerts(producer, scored)

            log.info(
                f"Tur #{round_num}: {n_total} satır | "
                f"IF anomali: {n_anom_if} | LSTM anomali: {n_anom_lstm} | "
                f"birleşik: {n_anom_combined} | alarm: {sent}"
            )

            if n_anom_combined > 0:
                top = scored[scored["is_anomaly"]].nsmallest(3, "if_score")
                log.info("En aykırı 3 satır:")
                for _, row in top.iterrows():
                    lstm_str = (
                        f"LSTM_MSE={row['lstm_mse']:.3f}"
                        if pd.notna(row["lstm_mse"])
                        else "LSTM_MSE=—"
                    )
                    log.info(
                        f"  {row['event_time']} | {row['city']:10s} | "
                        f"{(row['station_name'] or '')[:30]:30s} | "
                        f"PM25={row['pm25']:>5.1f} CO={row['co']:>5.1f} NO2={row['no2']:>5.1f} | "
                        f"IF={row['if_score']:.4f} | {lstm_str}"
                    )

            # State güncelle
            latest = scored["event_time"].max()
            state["last_processed_event_time"] = latest.isoformat()
            state["last_run_at"] = datetime.now(timezone.utc).isoformat()
            save_state(state)
        else:
            log.info(f"Tur #{round_num}: skorlanacak yeni satır yok.")

        # Bir sonraki tura kadar bekle (Ctrl+C duyarlı)
        if _shutdown:
            break
        log.info(f"Sonraki tur için {POLL_INTERVAL_SEC}s bekleniyor...")
        slept = 0.0
        while slept < POLL_INTERVAL_SEC and not _shutdown:
            time.sleep(min(1.0, POLL_INTERVAL_SEC - slept))
            slept += 1.0

    log.info("Scorer kapanıyor, Kafka flush ediliyor...")
    producer.flush()
    producer.close(timeout=10)
    log.info("Bye.")


if __name__ == "__main__":
    main()