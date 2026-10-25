import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from batch_layer.data_prep import prepare_dataset, ALL_FEATURES

CONTAMINATION = 0.05
N_ESTIMATORS  = 200
RANDOM_STATE  = 42

PROBE_START = "2026-05-11"
CUTOFFS = ["2026-05-01", "2026-05-03", "2026-05-05",
           "2026-05-07", "2026-05-09", "2026-05-11"]

_, _, full = prepare_dataset()
full = full.copy()
full["event_time"] = pd.to_datetime(full["event_time"], utc=True)
full = full.sort_values("event_time").reset_index(drop=True)

probe = full[full["event_time"] >= pd.Timestamp(PROBE_START, tz="UTC")]
print(f"Probe set: {len(probe)} satir  ({PROBE_START} sonrasi)")
print(f"Probe sehirleri: {dict(probe['city'].value_counts())}\n")
sonuclar = []
for c in CUTOFFS:
    slice_df = full[full["event_time"] < pd.Timestamp(c, tz="UTC")]
    if len(slice_df) < 50:
        print(f"{c}: yetersiz veri ({len(slice_df)}), atlandi")
        continue

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(slice_df[ALL_FEATURES].values)
    X_pr = scaler.transform(probe[ALL_FEATURES].values)

    m = IsolationForest(n_estimators=N_ESTIMATORS,
                        contamination=CONTAMINATION,
                        random_state=RANDOM_STATE, n_jobs=-1).fit(X_tr)

    s_tr = m.score_samples(X_tr)
    s_pr = m.score_samples(X_pr)
    tau  = float(np.percentile(s_tr, 100 * CONTAMINATION))

    sonuclar.append({
        "cutoff": c,
        "n_train": int(len(slice_df)),
        "n_cities": int(slice_df["city"].nunique()),
        "tau_if": tau,
        "train_anomaly_rate": float((m.predict(X_tr) == -1).mean()),
        "probe_anomaly_rate": float((s_pr <= tau).mean()),
        "probe_mean_score": float(s_pr.mean()),
    })
print(f"{'Cutoff':<12}{'nTrain':>7}{'Cities':>7}{'tau_IF':>9}"
      f"{'Train AR':>10}{'Probe AR':>10}{'Probe s̄':>10}")
print("-" * 65)
for r in sonuclar:
    print(f"{r['cutoff']:<12}{r['n_train']:>7}{r['n_cities']:>7}"
          f"{r['tau_if']:>9.4f}{r['train_anomaly_rate']:>9.2%}"
          f"{r['probe_anomaly_rate']:>10.2%}{r['probe_mean_score']:>10.4f}")

Path("data").mkdir(exist_ok=True)
json.dump({"probe_start": PROBE_START, "n_probe": int(len(probe)),
           "checkpoints": sonuclar},
          open("data/drift_replay_results.json", "w"), indent=2)
print("\nKaydedildi: data/drift_replay_results.json")
