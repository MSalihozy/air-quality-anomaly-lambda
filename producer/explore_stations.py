"""
OpenAQ keşif scripti.
Hedef şehirler için OpenAQ'da kaç istasyon, hangi parametreler var bul.
Sonuçları konsola bas + JSON dosyasına kaydet.
"""
import os
import json
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# --- Yapılandırma ---
load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAQ_API_KEY .env dosyasında bulunamadı")

BASE = "https://api.openaq.org/v3"
HEADERS = {"X-API-Key": API_KEY}

# Hedef şehirler — bbox koordinatlarıyla (her şehir merkezi etrafında ~30 km kare)
# bbox formatı: (minLon, minLat, maxLon, maxLat)
TARGET_CITIES = {
    # Türkiye
    "Istanbul":   (28.60, 40.85, 29.40, 41.20),
    "Canakkale":  (26.30, 40.05, 26.55, 40.20),
    # Avrupa başkentleri
    "Berlin":     (13.10, 52.35, 13.75, 52.65),
    "Paris":      ( 2.20, 48.80,  2.50, 48.95),
    "London":     (-0.30, 51.40,  0.10, 51.60),
    "Madrid":     (-3.85, 40.30, -3.55, 40.55),
    "Rome":       (12.35, 41.80, 12.60, 41.95),
    # Hindistan
    "Delhi":      (76.85, 28.45, 77.40, 28.90),
    "Mumbai":     (72.75, 18.90, 72.95, 19.30),
    "Kanpur":     (80.20, 26.35, 80.45, 26.55),
    "Kolkata":    (88.25, 22.45, 88.45, 22.65),
    # ABD
    "Washington_DC": (-77.15, 38.80, -76.90, 39.00),
    "Los_Angeles":   (-118.50, 33.90, -118.10, 34.20),
    "Houston":       (-95.55, 29.55, -95.20, 29.90),
    "Pittsburgh":    (-80.10, 40.35, -79.85, 40.55),
}

# İlgilendiğimiz parametreler (OpenAQ parameter id'leri)
TARGET_PARAMS = {2: "pm25", 1: "pm10", 7: "no2", 8: "co", 9: "so2", 10: "o3"}


def fetch_locations_for_bbox(bbox):
    """Verilen bbox içindeki tüm OpenAQ lokasyonlarını döndür."""
    minLon, minLat, maxLon, maxLat = bbox
    url = f"{BASE}/locations"
    params = {
        "bbox": f"{minLon},{minLat},{maxLon},{maxLat}",
        "limit": 1000,
    }
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("results", [])


def summarize_location(loc):
    """Bir lokasyon objesini sade dict'e indirge."""
    sensors = loc.get("sensors", []) or []
    params_in_station = []
    for s in sensors:
        p = (s.get("parameter") or {})
        pid = p.get("id")
        pname = p.get("name")
        if pid in TARGET_PARAMS:
            params_in_station.append({
                "sensor_id": s.get("id"),
                "parameter_id": pid,
                "parameter_name": pname,
                "units": p.get("units"),
            })
    coords = loc.get("coordinates") or {}
    return {
        "location_id": loc.get("id"),
        "name": loc.get("name"),
        "country": (loc.get("country") or {}).get("code"),
        "latitude": coords.get("latitude"),
        "longitude": coords.get("longitude"),
        "is_mobile": loc.get("isMobile"),
        "sensors": params_in_station,
        "datetime_last": (loc.get("datetimeLast") or {}).get("utc"),
    }


def main():
    out_dir = Path("data/raw_parquet")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    summary_rows = []

    for city, bbox in TARGET_CITIES.items():
        print(f"\n=== {city} ===")
        try:
            locs = fetch_locations_for_bbox(bbox)
        except requests.HTTPError as e:
            print(f"  HATA: {e}")
            continue

        # Hedef parametrelerden en az birini ölçen istasyonları filtrele
        relevant = []
        for loc in locs:
            summ = summarize_location(loc)
            if summ["sensors"]:  # boşsa istasyonun bizim parametrelerle alakası yok
                relevant.append(summ)

        all_results[city] = relevant

        # Özet — kaç istasyon, hangi parametreler kapsanıyor
        param_set = set()
        for r in relevant:
            for s in r["sensors"]:
                param_set.add(s["parameter_name"])
        print(f"  Toplam istasyon: {len(locs)}, hedef parametre içeren: {len(relevant)}")
        print(f"  Kapsanan parametreler: {sorted(param_set) if param_set else 'YOK'}")

        # İlk 3 istasyonun adını yazdır
        for r in relevant[:3]:
            param_names = [s["parameter_name"] for s in r["sensors"]]
            name_short = (r["name"] or "")[:40]
            print(f"    - id={r['location_id']:>8} {name_short:40s} [{','.join(param_names)}] last={r['datetime_last']}")

        summary_rows.append({
            "city": city,
            "total_locations": len(locs),
            "relevant_locations": len(relevant),
            "parameters": sorted(param_set),
        })

        # OpenAQ rate limit (saatte 60 istek, dakikada da var). 15 şehir için 1 saniye yeterli.
        time.sleep(1)

    # Sonuçları kaydet
    out_path = out_dir / "stations_metadata.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n\nMetadata kaydedildi: {out_path}")

    print("\n=== ÖZET ===")
    for row in summary_rows:
        print(f"  {row['city']:18s} | toplam: {row['total_locations']:4d} | hedef: {row['relevant_locations']:4d} | params: {row['parameters']}")


if __name__ == "__main__":
    main()