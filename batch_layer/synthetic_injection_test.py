import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from pathlib import Path
from datetime import datetime, timezone

from batch_layer.data_prep import (
    long_to_wide,
    add_engineered_features,
    filter_complete_rows,
    impute_missing,
    CORE_FEATURES,
    ALL_FEATURES,
)

# --- Sabitler ---
RAW_EVENTS_PATH = "data/raw_parquet/events"
IF_MODEL_PATH = Path("data/models/isolation_forest.joblib")
IF_SCALER_PATH = Path("data/models/if_scaler.joblib")
LSTM_MODEL_PATH = Path("data/models/lstm_autoencoder_full.keras")
LSTM_SCALER_PATH = Path("data/models/lstm_scaler_full.joblib")
LSTM_META_PATH = Path("data/models/lstm_metadata_full.json")

N_PER_TYPE = 20
SEQUENCE_LEN = 12

np.random.seed(42)


# ====================================================================
def load_models():
    print("[1/6] Modeller yükleniyor...")
    if_model = joblib.load(IF_MODEL_PATH)
    if_scaler = joblib.load(IF_SCALER_PATH)
    lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH, compile=False)
    lstm_scaler = joblib.load(LSTM_SCALER_PATH)
    with open(LSTM_META_PATH) as f:
        lstm_meta = json.load(f)
    lstm_threshold = lstm_meta["threshold_p99"]
    print(f"      ✓ IF, LSTM-AE yüklendi. LSTM eşik p99 = {lstm_threshold:.4f}")
    return if_model, if_scaler, lstm_model, lstm_scaler, lstm_threshold


def load_clean_data():
    print("[2/6] Temiz veri kümesi hazırlanıyor...")
    df_long = pd.read_parquet(RAW_EVENTS_PATH)
    df_long["event_time"] = pd.to_datetime(df_long["event_time"], utc=True)
    df_wide = long_to_wide(df_long)
    df_wide = add_engineered_features(df_wide)
    df_wide = filter_complete_rows(df_wide, min_features=8)
    df_wide = impute_missing(df_wide)
    df_wide = df_wide.sort_values(["station_id", "event_time"]).reset_index(drop=True)
    print(f"      ✓ {len(df_wide)} satır temiz veri hazır.")
    return df_wide


# ====================================================================
def score_if_row(row, if_model, if_scaler):
    X = row[ALL_FEATURES].values.reshape(1, -1)
    Xs = if_scaler.transform(X)
    pred = if_model.predict(Xs)[0]
    return pred == -1


def score_lstm_sequence(sequence, lstm_model, lstm_scaler, threshold):
    seq_arr = sequence[CORE_FEATURES].values
    seq_scaled = lstm_scaler.transform(seq_arr)
    seq_input = seq_scaled.reshape(1, SEQUENCE_LEN, len(CORE_FEATURES))
    seq_recon = lstm_model.predict(seq_input, verbose=0)
    mse = np.mean(np.square(seq_input - seq_recon))
    return mse > threshold


# ====================================================================
# ENJEKSİYON FONKSİYONLARI — SUBTLE
# ====================================================================
def inject_spike_subtle(seq):
    """PM2.5 ve PM10 son saatte × 5."""
    seq_inj = seq.copy()
    last_idx = seq_inj.index[-1]
    seq_inj.loc[last_idx, "pm25"] = seq_inj.loc[last_idx, "pm25"] * 5.0
    seq_inj.loc[last_idx, "pm10"] = seq_inj.loc[last_idx, "pm10"] * 5.0
    return seq_inj


def inject_drift_subtle(seq, step_pct=0.20):
    """%20/saat kademeli artış."""
    seq_inj = seq.copy()
    base_pm25 = seq_inj["pm25"].iloc[0]
    base_pm10 = seq_inj["pm10"].iloc[0]
    multipliers = np.array([(1 + step_pct) ** i for i in range(SEQUENCE_LEN)])
    seq_inj["pm25"] = base_pm25 * multipliers
    seq_inj["pm10"] = base_pm10 * multipliers
    return seq_inj


def inject_flatline_subtle(seq):
    """Sadece PM2.5/PM10 sabit (diğerleri normal)."""
    seq_inj = seq.copy()
    seq_inj["pm25"] = seq_inj["pm25"].iloc[0]
    seq_inj["pm10"] = seq_inj["pm10"].iloc[0]
    return seq_inj


# ====================================================================
# ENJEKSİYON FONKSİYONLARI — PRONOUNCED
# ====================================================================
def inject_spike_pronounced(seq):
    """PM2.5 = 500 μg/m³, PM10 = 800 μg/m³ (WHO eşik ötesi, ama tavan altında)."""
    seq_inj = seq.copy()
    last_idx = seq_inj.index[-1]
    seq_inj.loc[last_idx, "pm25"] = 500.0
    seq_inj.loc[last_idx, "pm10"] = 800.0
    return seq_inj


def inject_drift_pronounced(seq, step_pct=0.50):
    """%50/saat kademeli artış — çok agresif."""
    seq_inj = seq.copy()
    base_pm25 = seq_inj["pm25"].iloc[0]
    base_pm10 = seq_inj["pm10"].iloc[0]
    multipliers = np.array([(1 + step_pct) ** i for i in range(SEQUENCE_LEN)])
    seq_inj["pm25"] = base_pm25 * multipliers
    seq_inj["pm10"] = base_pm10 * multipliers
    return seq_inj


def inject_flatline_pronounced(seq):
    """Tüm 10 özellik 12 saat boyunca sabit (sensör tam donmuş)."""
    seq_inj = seq.copy()
    for feat in CORE_FEATURES:
        seq_inj[feat] = seq_inj[feat].iloc[0]
    return seq_inj


# ====================================================================
# TEST RUNNER
# ====================================================================
def run_injection_set(sequences, inject_func, if_model, if_scaler, lstm_model, lstm_scaler, lstm_threshold):
    """Bir set diziye enjeksiyon uygula ve sonuçları topla."""
    rows = []
    for seq in sequences:
        # Temiz
        clean_if = score_if_row(seq.iloc[-1], if_model, if_scaler)
        clean_lstm = score_lstm_sequence(seq, lstm_model, lstm_scaler, lstm_threshold)
        # Enjekte
        inj_seq = inject_func(seq)
        inj_if = score_if_row(inj_seq.iloc[-1], if_model, if_scaler)
        inj_lstm = score_lstm_sequence(inj_seq, lstm_model, lstm_scaler, lstm_threshold)
        rows.append({
            "clean_if": bool(clean_if), "clean_lstm": bool(clean_lstm),
            "inj_if": bool(inj_if), "inj_lstm": bool(inj_lstm),
        })
    return rows


def calc_metrics(rows, key_clean, key_inj):
    TP = sum(1 for r in rows if r[key_inj])
    FN = sum(1 for r in rows if not r[key_inj])
    FP = sum(1 for r in rows if r[key_clean])
    TN = sum(1 for r in rows if not r[key_clean])
    p = TP / (TP + FP) if (TP + FP) > 0 else 0
    r_ = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * p * r_ / (p + r_) if (p + r_) > 0 else 0
    return {"TP": TP, "FN": FN, "FP": FP, "TN": TN,
            "precision": p, "recall": r_, "f1": f1}


def calc_union(rows):
    TP = sum(1 for r in rows if r["inj_if"] or r["inj_lstm"])
    FN = sum(1 for r in rows if not (r["inj_if"] or r["inj_lstm"]))
    FP = sum(1 for r in rows if r["clean_if"] or r["clean_lstm"])
    TN = sum(1 for r in rows if not (r["clean_if"] or r["clean_lstm"]))
    p = TP / (TP + FP) if (TP + FP) > 0 else 0
    r_ = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * p * r_ / (p + r_) if (p + r_) > 0 else 0
    return {"TP": TP, "FN": FN, "FP": FP, "TN": TN,
            "precision": p, "recall": r_, "f1": f1}


def print_set_results(set_name, results_dict):
    print(f"\n{'='*78}")
    print(f"{set_name.upper()} ANOMALILER")
    print(f"{'='*78}")
    for anom_type in ["spike", "drift", "flatline"]:
        rows = results_dict[anom_type]
        if_m = calc_metrics(rows, "clean_if", "inj_if")
        lstm_m = calc_metrics(rows, "clean_lstm", "inj_lstm")
        union_m = calc_union(rows)
        print(f"\n--- {anom_type.upper()} ({len(rows)} örneklem) ---")
        print(f"{'Model':<22} {'TP':>4} {'FN':>4} {'FP':>4} {'TN':>4} "
              f"{'Prec':>7} {'Rec':>7} {'F1':>7}")
        print(f"{'Isolation Forest':<22} {if_m['TP']:>4} {if_m['FN']:>4} "
              f"{if_m['FP']:>4} {if_m['TN']:>4} "
              f"{if_m['precision']:>6.3f}  {if_m['recall']:>6.3f}  {if_m['f1']:>6.3f}")
        print(f"{'LSTM-AE':<22} {lstm_m['TP']:>4} {lstm_m['FN']:>4} "
              f"{lstm_m['FP']:>4} {lstm_m['TN']:>4} "
              f"{lstm_m['precision']:>6.3f}  {lstm_m['recall']:>6.3f}  {lstm_m['f1']:>6.3f}")
        print(f"{'Hibrit (Birleşim)':<22} {union_m['TP']:>4} {union_m['FN']:>4} "
              f"{union_m['FP']:>4} {union_m['TN']:>4} "
              f"{union_m['precision']:>6.3f}  {union_m['recall']:>6.3f}  {union_m['f1']:>6.3f}")


# ====================================================================
# ANA TEST
# ====================================================================
def run_test():
    if_model, if_scaler, lstm_model, lstm_scaler, lstm_threshold = load_models()
    df = load_clean_data()

    station_counts = df.groupby("station_id").size()
    valid_stations = station_counts[station_counts >= SEQUENCE_LEN].index.tolist()
    print(f"      ✓ {len(valid_stations)} istasyon ≥{SEQUENCE_LEN} saatlik veri içeriyor.")

    print(f"[3/6] {N_PER_TYPE * 3} adet dizi rastgele seçiliyor...")
    selected_sequences = []
    attempts = 0
    while len(selected_sequences) < N_PER_TYPE * 3 and attempts < 1000:
        station = np.random.choice(valid_stations)
        station_df = df[df["station_id"] == station].reset_index(drop=True)
        if len(station_df) < SEQUENCE_LEN:
            attempts += 1
            continue
        start_idx = np.random.randint(0, len(station_df) - SEQUENCE_LEN + 1)
        seq = station_df.iloc[start_idx:start_idx + SEQUENCE_LEN].copy()
        if len(seq) == SEQUENCE_LEN:
            selected_sequences.append(seq)
        attempts += 1

    print(f"      ✓ {len(selected_sequences)} dizi seçildi.")

    spike_seqs = selected_sequences[:N_PER_TYPE]
    drift_seqs = selected_sequences[N_PER_TYPE:N_PER_TYPE * 2]
    flatline_seqs = selected_sequences[N_PER_TYPE * 2:N_PER_TYPE * 3]

    # ============================================================
    # SUBTLE
    # ============================================================
    print("[4/6] SUBTLE anomaliler enjekte ediliyor ve skorlanıyor...")
    subtle_results = {
        "spike": run_injection_set(spike_seqs, inject_spike_subtle,
                                    if_model, if_scaler, lstm_model, lstm_scaler, lstm_threshold),
        "drift": run_injection_set(drift_seqs, inject_drift_subtle,
                                    if_model, if_scaler, lstm_model, lstm_scaler, lstm_threshold),
        "flatline": run_injection_set(flatline_seqs, inject_flatline_subtle,
                                       if_model, if_scaler, lstm_model, lstm_scaler, lstm_threshold),
    }

    # ============================================================
    # PRONOUNCED
    # ============================================================
    print("[5/6] PRONOUNCED anomaliler enjekte ediliyor ve skorlanıyor...")
    pronounced_results = {
        "spike": run_injection_set(spike_seqs, inject_spike_pronounced,
                                    if_model, if_scaler, lstm_model, lstm_scaler, lstm_threshold),
        "drift": run_injection_set(drift_seqs, inject_drift_pronounced,
                                    if_model, if_scaler, lstm_model, lstm_scaler, lstm_threshold),
        "flatline": run_injection_set(flatline_seqs, inject_flatline_pronounced,
                                       if_model, if_scaler, lstm_model, lstm_scaler, lstm_threshold),
    }

    # ============================================================
    print("[6/6] Sonuç raporları üretiliyor...")
    print_set_results("SUBTLE", subtle_results)
    print_set_results("PRONOUNCED", pronounced_results)

    # JSON kaydet
    def serialize(rd):
        return {k: {
            "isolation_forest": calc_metrics(v, "clean_if", "inj_if"),
            "lstm_ae": calc_metrics(v, "clean_lstm", "inj_lstm"),
            "hybrid_union": calc_union(v),
        } for k, v in rd.items()}

    output_data = {
        "tested_at": datetime.now(timezone.utc).isoformat(),
        "n_per_type": N_PER_TYPE,
        "lstm_threshold_p99": lstm_threshold,
        "subtle": serialize(subtle_results),
        "pronounced": serialize(pronounced_results),
    }
    output_path = Path("data/synthetic_test_results.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n✓ Sonuçlar kaydedildi: {output_path}")


if __name__ == "__main__":
    run_test()