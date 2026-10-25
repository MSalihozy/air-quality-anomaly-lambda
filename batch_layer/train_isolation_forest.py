"""
Batch Layer — Isolation Forest eğitimi.

data_prep.py'nin train/val DataFrame'lerini alır, IF eğitir, model + scaler
+ feature listesini joblib ile data/models/ altına kaydeder.

Çalıştırma:
  python -m batch_layer.train_isolation_forest
"""
import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from batch_layer.data_prep import prepare_dataset, ALL_FEATURES, CORE_FEATURES

MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

IF_MODEL_PATH = MODELS_DIR / "isolation_forest.joblib"
SCALER_PATH   = MODELS_DIR / "if_scaler.joblib"
META_PATH     = MODELS_DIR / "if_metadata.json"

# Hiperparametreler
CONTAMINATION = 0.05      # Verinin %5'i anomali varsayımı
N_ESTIMATORS  = 200
RANDOM_STATE  = 42


def fit_isolation_forest(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """IF + scaler eğit, val üzerinde değerlendir, dosyaya kaydet."""
    feature_cols = ALL_FEATURES  # CORE + engineered

    # 1) Scale (IF zorunlu değil ama distance-based metrics için iyi pratik)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_val   = scaler.transform(val_df[feature_cols].values)

    # 2) Modeli eğit
    print(f"[IF] Eğitim başlıyor — n_train={len(X_train)}, features={len(feature_cols)}, contamination={CONTAMINATION}")
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train)
    print("[IF] Eğitim tamamlandı.")

    # 3) Train ve val tahminleri
    train_pred  = model.predict(X_train)        # +1 normal, -1 anomali
    val_pred    = model.predict(X_val)
    train_score = model.score_samples(X_train)  # büyük → daha normal
    val_score   = model.score_samples(X_val)

    # IF -1 etiketini True (anomali) olarak çevir
    train_anomaly_rate = float((train_pred == -1).mean())
    val_anomaly_rate   = float((val_pred == -1).mean())

    print(f"[IF] Train anomali oranı: {train_anomaly_rate:.2%}")
    print(f"[IF] Val   anomali oranı: {val_anomaly_rate:.2%}")

    # En anormal 5 train satırını göster (sanity check)
    train_with_score = train_df.copy()
    train_with_score["if_score"] = train_score
    train_with_score["if_pred_anomaly"] = (train_pred == -1)
    top_anomalies = train_with_score.nsmallest(5, "if_score")
    print("\n[IF] En anormal 5 train satırı:")
    print(top_anomalies[
        ["event_time", "city", "station_name", "pm25", "co", "no2", "if_score", "if_pred_anomaly"]
    ].to_string(index=False))

    # 4) Modeli kaydet
    joblib.dump(model, IF_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train_samples": int(len(X_train)),
        "n_val_samples": int(len(X_val)),
        "feature_columns": feature_cols,
        "core_features": CORE_FEATURES,
        "hyperparameters": {
            "n_estimators": N_ESTIMATORS,
            "contamination": CONTAMINATION,
            "random_state": RANDOM_STATE,
        },
        "train_anomaly_rate": train_anomaly_rate,
        "val_anomaly_rate": val_anomaly_rate,
        "train_score_stats": {
            "mean": float(train_score.mean()),
            "std":  float(train_score.std()),
            "p01":  float(np.percentile(train_score, 1)),
            "p05":  float(np.percentile(train_score, 5)),
            "p50":  float(np.percentile(train_score, 50)),
        },
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[IF] Model kaydedildi: {IF_MODEL_PATH}")
    print(f"[IF] Scaler kaydedildi: {SCALER_PATH}")
    print(f"[IF] Metadata kaydedildi: {META_PATH}")

    return model, scaler, metadata


def main():
    print("=" * 60)
    print("ISOLATION FOREST TRAINING")
    print("=" * 60)
    train, val, _ = prepare_dataset()
    fit_isolation_forest(train, val)


if __name__ == "__main__":
    main()