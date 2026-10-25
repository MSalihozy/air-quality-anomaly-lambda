"""
AQICN producer — keşif scriptinin ürettiği aqicn_metadata.json'ı okur,
her istasyon için /feed/@uid/ endpoint'ini polling ile çağırır,
gelen ölçümleri normalize edilmiş JSON event'i olarak Kafka'ya yollar.

Test modu: 4 şehir alt kümesi, 2 dakikada bir tur.
"""
import os
import json
import time
import signal
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from kafka import KafkaProducer
from kafka.errors import KafkaError

# --- Yapılandırma ---
load_dotenv()
TOKEN = os.getenv("AQICN_API_TOKEN")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC_AIR", "air_quality_events")

if not TOKEN:
    raise RuntimeError("AQICN_API_TOKEN .env dosyasında bulunamadı")

BASE = "https://api.waqi.info"
METADATA_PATH = Path("data/raw_parquet/aqicn_metadata.json")

# Test modu: sadece bu 4 şehirle başlıyoruz
# Tüm şehirlere yayılıyoruz — full kapsam
TEST_CITIES = {
    "Istanbul", "Ankara", "Antalya", "Erzurum", "Konya", "Bursa",
    "Berlin", "Paris", "London",
    "Delhi", "Mumbai",
    "Los_Angeles", "Houston",
}
POLL_INTERVAL_SEC = 300        # 5 dakika (rate limit nezaketi)
PER_REQUEST_SLEEP = 0.4
HTTP_TIMEOUT = 30

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aqicn-producer")

# Graceful shutdown için flag
_shutdown = False
def _handle_sigint(signum, frame):
    global _shutdown
    log.info("Shutdown sinyali alındı, mevcut tur bitince çıkacak.")
    _shutdown = True
signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)


# --- Yardımcılar ---
def load_target_stations():
    """Metadata JSON'dan test şehirlerinin istasyonlarını çıkar."""
    if not METADATA_PATH.exists():
        raise RuntimeError(f"Metadata yok: {METADATA_PATH} — önce explore_aqicn.py çalıştır")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    stations = []
    for city, payload in meta.items():
        if city not in TEST_CITIES:
            continue
        # explore_aqicn.py sadece her şehir için 3 örneklem detay tuttu;
        # producer'da bu örneklem yeterli (tüm istasyonu polling rate limit'i zorlar)
        for st in payload.get("sample_details", []):
            stations.append({
                "city": city,
                "uid": st["uid"],
                "name": st["name"],
                "lat": st["lat"],
                "lon": st["lon"],
            })
    return stations


def fetch_feed(uid):
    """Tek istasyonun anlık feed'ini al."""
    url = f"{BASE}/feed/@{uid}/"
    resp = requests.get(url, params={"token": TOKEN}, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def normalize_event(city, station_meta, feed_payload):
    """
    AQICN feed'ini düz event şemasına çevir.
    Bir istasyonun feed cevabı içinde her parametre ayrı bir alanda; biz
    `parameter` ve `value` olacak şekilde DAR uzun (long) format kullanıyoruz.
    Yani tek istasyonun tek polling'inden N parametre = N event çıkar.
    """
    if feed_payload.get("status") != "ok":
        return []

    data = feed_payload.get("data", {}) or {}
    iaqi = data.get("iaqi", {}) or {}
    time_info = data.get("time", {}) or {}
    measured_iso = time_info.get("iso") or time_info.get("s")
    aqi = data.get("aqi")  # genel AQI değeri (None olabilir)

    now = datetime.now(timezone.utc).isoformat()
    base_event = {
        "source": "aqicn",
        "city": city,
        "station_id": str(station_meta["uid"]),
        "station_name": station_meta["name"],
        "lat": station_meta["lat"],
        "lon": station_meta["lon"],
        "measured_at": measured_iso,    # event_time olarak kullanılacak
        "ingested_at": now,             # processing_time olarak kullanılacak
        "aqi": aqi if isinstance(aqi, (int, float)) else None,
    }

    events = []
    for param_name, param_payload in iaqi.items():
        # iaqi şu yapıda: {"pm25": {"v": 23.4}, "no2": {"v": 12.1}, ...}
        if not isinstance(param_payload, dict):
            continue
        value = param_payload.get("v")
        if value is None:
            continue
        ev = dict(base_event)
        ev["parameter"] = param_name
        ev["value"] = float(value)
        # Her parametre+istasyon+ölçüm-zamanı kombinasyonu için unique key
        ev["event_id"] = f"aqicn:{station_meta['uid']}:{param_name}:{measured_iso}"
        events.append(ev)
    return events


def make_producer():
    """Kafka producer factory — JSON serializer + idempotence ayarlı."""
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks="all",
        retries=5,
        linger_ms=50,
    )


# --- Ana döngü ---
def main():
    stations = load_target_stations()
    log.info(f"Hedef istasyon sayısı: {len(stations)}")
    for s in stations:
        log.info(f"  - {s['city']:10s} | uid={s['uid']:>7} | {s['name']}")

    producer = make_producer()
    log.info(f"Kafka producer hazır → {KAFKA_BOOTSTRAP} / topic={TOPIC}")

    round_num = 0
    while not _shutdown:
        round_num += 1
        log.info(f"=== Tur #{round_num} başlıyor ===")
        round_start = time.time()
        sent_total = 0
        failed_stations = 0

        for st in stations:
            if _shutdown:
                break
            try:
                feed = fetch_feed(st["uid"])
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
                log.warning(f"Feed HATA city={st['city']} uid={st['uid']}: {e}")
                failed_stations += 1
                time.sleep(PER_REQUEST_SLEEP)
                continue

            events = normalize_event(st["city"], st, feed)
            if not events:
                log.warning(f"Boş feed: city={st['city']} uid={st['uid']}")

            for ev in events:
                try:
                    producer.send(
                        TOPIC,
                        key=ev["station_id"],   # aynı istasyon hep aynı partition'a gitsin
                        value=ev,
                    )
                    sent_total += 1
                except KafkaError as e:
                    log.error(f"Kafka send hata: {e}")

            time.sleep(PER_REQUEST_SLEEP)

        producer.flush()
        elapsed = time.time() - round_start
        log.info(
            f"Tur #{round_num} bitti | gönderilen event: {sent_total} | "
            f"hatalı istasyon: {failed_stations} | süre: {elapsed:.1f}s"
        )

        # Bir sonraki tura kadar bekle (shutdown sinyaline duyarlı)
        if _shutdown:
            break
        wait_remaining = max(0.0, POLL_INTERVAL_SEC - elapsed)
        log.info(f"Sonraki tur için {wait_remaining:.0f}s bekleniyor...")
        # 1-saniyelik chunklarla bekle ki Ctrl+C anında yanıt versin
        slept = 0.0
        while slept < wait_remaining and not _shutdown:
            time.sleep(min(1.0, wait_remaining - slept))
            slept += 1.0

    log.info("Producer kapanıyor, Kafka flush ediliyor...")
    producer.flush()
    producer.close(timeout=10)
    log.info("Bye.")


if __name__ == "__main__":
    main()