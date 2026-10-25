"""
AQICN keşif scripti.
Hedef şehirler için AQICN'de kaç istasyon, hangi parametreler var, ne kadar güncel?
Sonuçları konsola bas + JSON dosyasına kaydet.

AQICN endpoint'leri:
  /v2/map/bounds/?latlng=minLat,minLng,maxLat,maxLng&token=...  → bbox içindeki istasyonlar
  /feed/@stationId/?token=...                                    → tek istasyon detayı
"""
import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("AQICN_API_TOKEN")
if not TOKEN:
    raise RuntimeError("AQICN_API_TOKEN .env dosyasında bulunamadı")

BASE = "https://api.waqi.info"

# Hedef şehirler — AQICN için latlng formatı: (minLat, minLng, maxLat, maxLng)
# (OpenAQ bbox sırası farklıydı: minLon, minLat, maxLon, maxLat)
TARGET_CITIES = {
    # Türkiye (6)
    "Istanbul":   (40.85,  28.60, 41.20,  29.40),
    "Ankara":     (39.85,  32.65, 40.05,  33.00),
    "Antalya":    (36.80,  30.55, 37.00,  30.85),
    "Erzurum":    (39.85,  41.20, 39.98,  41.40),
    "Konya":      (37.80,  32.40, 37.95,  32.65),
    "Bursa":      (40.15,  28.95, 40.30,  29.20),
    # Avrupa (3)
    "Berlin":     (52.35,  13.10, 52.65,  13.75),
    "Paris":      (48.80,   2.20, 48.95,   2.50),
    "London":     (51.40,  -0.30, 51.60,   0.10),
    # Hindistan (2)
    "Delhi":      (28.45,  76.85, 28.90,  77.40),
    "Mumbai":     (18.90,  72.75, 19.30,  72.95),
    # ABD (2) — Houston ve LA orta kirlilik + sanayi profili
    "Los_Angeles":   (33.90, -118.50, 34.20, -118.10),
    "Houston":       (29.55,  -95.55, 29.90,  -95.20),
}


def fetch_stations_in_bbox(bbox):
    """latlng bbox içindeki AQICN istasyonlarını döndür (özet bilgi)."""
    minLat, minLng, maxLat, maxLng = bbox
    url = f"{BASE}/v2/map/bounds/"
    params = {
        "latlng": f"{minLat},{minLng},{maxLat},{maxLng}",
        "token": TOKEN,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "ok":
        return [], payload.get("data", "no data")
    return payload.get("data", []), None


def fetch_station_detail(uid):
    """Tek istasyonun detayını al — IAQI (her parametrenin AQI değeri) burada gelir."""
    url = f"{BASE}/feed/@{uid}/"
    params = {"token": TOKEN}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "ok":
        return None
    return payload.get("data", {})


def parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def main():
    out_dir = Path("data/raw_parquet")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    summary_rows = []
    now = datetime.now(timezone.utc)

    for city, bbox in TARGET_CITIES.items():
        print(f"\n=== {city} ===")
        try:
            stations, err = fetch_stations_in_bbox(bbox)
        except requests.HTTPError as e:
            print(f"  HATA: {e}")
            continue

        if err:
            print(f"  API uyarı: {err}")

        print(f"  Bölgedeki istasyon sayısı: {len(stations)}")

        # İlk 3 istasyonun detayını çek (parametre + zaman bilgisi için)
        # Hepsi için yapsak rate limit dolar; örneklem yeterli keşif için
        sample_details = []
        for st in stations[:3]:
            uid = st.get("uid")
            try:
                detail = fetch_station_detail(uid)
            except requests.HTTPError as e:
                print(f"  Detay HATA (uid={uid}): {e}")
                continue
            if not detail:
                continue
            iaqi = detail.get("iaqi", {}) or {}
            params_in_station = sorted(iaqi.keys())
            time_info = detail.get("time", {}) or {}
            measured_at = time_info.get("iso") or time_info.get("s")
            sample_details.append({
                "uid": uid,
                "name": (st.get("station") or {}).get("name"),
                "lat": st.get("lat"),
                "lon": st.get("lon"),
                "aqi": st.get("aqi"),
                "parameters": params_in_station,
                "measured_at": measured_at,
            })
            time.sleep(0.5)  # her detay çağrısı arasında küçük bekleme

        all_results[city] = {
            "bbox_count": len(stations),
            "sample_details": sample_details,
        }

        # Konsola örnek istasyonları yaz
        param_set = set()
        for d in sample_details:
            param_set.update(d["parameters"])
            name_short = (d["name"] or "")[:40]
            params_str = ",".join(d["parameters"]) if d["parameters"] else "—"
            print(f"    - uid={d['uid']:>6} {name_short:40s} AQI={d['aqi']:>4} [{params_str}] @ {d['measured_at']}")

        print(f"  Örneklemde gözlenen parametreler: {sorted(param_set) if param_set else 'YOK'}")

        summary_rows.append({
            "city": city,
            "stations_in_bbox": len(stations),
            "sampled": len(sample_details),
            "params_seen": sorted(param_set),
        })

        time.sleep(1)

    # Sonuçları kaydet
    out_path = out_dir / "aqicn_metadata.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n\nMetadata kaydedildi: {out_path}")

    print("\n=== ÖZET ===")
    for row in summary_rows:
        print(f"  {row['city']:18s} | bbox: {row['stations_in_bbox']:4d} | örneklem: {row['sampled']:2d} | params: {row['params_seen']}")


if __name__ == "__main__":
    main()