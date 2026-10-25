"""
Regenerates every dataset-level number reported in the paper
from the frozen snapshot. Run: python -m tools.verify_paper_numbers
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from batch_layer.data_prep import prepare_dataset, CORE_FEATURES
from batch_layer.train_lstm_ae import make_sequences

TURKEY = {"Istanbul","Ankara","Antalya","Erzurum","Konya","Bursa","Canakkale"}

def kontrol(ad, beklenen, bulunan):
    isaret = "OK " if beklenen == bulunan else "FARK"
    print(f"  [{isaret}] {ad:<42} paper={beklenen:<8} computed={bulunan}")
    return beklenen == bulunan

print("="*78)
print("PAPER NUMBER VERIFICATION")
print("="*78)

snap = pd.read_parquet("data/snapshot/aqicn_events_snapshot.parquet")
train, val, full = prepare_dataset()

sonuclar = []
print("\n-- Table 6 / Table 7: ingestion and filtering --")
sonuclar.append(kontrol("Total raw measurements ingested", 20225, len(snap)))
sonuclar.append(kontrol("Monitoring stations", 39, snap["station_id"].nunique()))
sonuclar.append(kontrol("Filtered valid station-hour records", 1333, len(full)))
sonuclar.append(kontrol("IF training rows", 1066, len(train)))
sonuclar.append(kontrol("IF validation rows", 267, len(val)))
sonuclar.append(kontrol("Stations passing feature filter", 26,
                        full["station_id"].nunique()))

print("\n-- Table 10 / Section 6.3: sequence counts --")
tr_noTR = train[~train["city"].isin(TURKEY)].reset_index(drop=True)
val_noTR = val[~val["city"].isin(TURKEY)].reset_index(drop=True)
Xa, _ = make_sequences(tr_noTR, CORE_FEATURES, 12)
Xb, _ = make_sequences(train,   CORE_FEATURES, 12)
Xv, _ = make_sequences(val,     CORE_FEATURES, 12)
Xvn, _ = make_sequences(val_noTR, CORE_FEATURES, 12)

sonuclar.append(kontrol("Version A training sequences", 347, len(Xa)))
sonuclar.append(kontrol("Version B training sequences", 793, len(Xb)))
sonuclar.append(kontrol("Validation sequences", 41, len(Xv)))
sonuclar.append(kontrol("Validation sequences (Turkey excluded)", 41, len(Xvn)))

print("\n-- Table 14: Mumbai Powai saturation --")
snap["param"] = snap["event_id"].astype(str).str.split(":").str[2]
powai = snap[snap["station_name"].astype(str).str.contains("Powai", na=False)]
pm25 = powai.loc[powai["param"]=="pm25", "value"].dropna()
pm10 = powai.loc[powai["param"]=="pm10", "value"].dropna()
sonuclar.append(kontrol("PM2.5 records at ceiling (758)", 34, int((pm25>=758).sum())))
sonuclar.append(kontrol("PM10 records at ceiling (999)", 33, int((pm10>=999).sum())))

print("\n-- Section 4.4: saturation threshold justification --")
sonuclar.append(kontrol("PM2.5 readings in [500, 758)", 0,
                        int(((pm25>=500)&(pm25<758)).sum())))
all_pm25 = snap.loc[snap["param"]=="pm25", "value"].dropna()
sonuclar.append(kontrol("Highest genuine PM2.5 (below ceiling)", 298,
                        int(all_pm25[all_pm25<758].max())))

print("\n-- Model artefacts --")
lb = json.load(open("data/models/lstm_metadata_full.json"))
sonuclar.append(kontrol("Version B epochs completed", 156, lb["epochs_run"]))
sonuclar.append(kontrol("tau_LSTM (p99)", 0.791, round(lb["threshold_p99"],3)))
ifm = json.load(open("data/models/if_metadata.json"))
sonuclar.append(kontrol("tau_IF (p05 of train scores)", -0.549,
                        round(ifm["train_score_stats"]["p05"],3)))

print("\n" + "="*78)
print(f"RESULT: {sum(sonuclar)}/{len(sonuclar)} checks passed")
print("="*78)
sys.exit(0 if all(sonuclar) else 1)
