"""
Batch Layer — LSTM Autoencoder eğitimi (FULL — 13 şehir tamamı dahil).

Versiyon A (train_lstm_ae.py) Türkiye istasyonlarını hariç tutuyor.
Bu versiyon TÜM 13 şehri dahil ediyor — Türkiye dahil.
Karşılaştırma amaçlı: küçük örnek bir bölgenin (Türkiye) modelin
genelleme yeteneğine etkisi.

Modeller _full suffix'i ile kaydedilir, Versiyon A'nın üzerine yazılmaz.

Çalıştırma:
  python -m batch_layer.train_lstm_ae_full
"""
import os
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

Model = tf.keras.Model
Input = tf.keras.layers.Input
LSTM = tf.keras.layers.LSTM
RepeatVector = tf.keras.layers.RepeatVector
TimeDistributed = tf.keras.layers.TimeDistributed
Dense = tf.keras.layers.Dense
EarlyStopping = tf.keras.callbacks.EarlyStopping

from sklearn.preprocessing import StandardScaler

from batch_layer.data_prep import prepare_dataset, CORE_FEATURES
from batch_layer.train_lstm_ae import make_sequences, build_lstm_autoencoder

MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Versiyon B — _full suffix
LSTM_MODEL_PATH = MODELS_DIR / "lstm_autoencoder_full.keras"
LSTM_SCALER_PATH = MODELS_DIR / "lstm_scaler_full.joblib"
LSTM_META_PATH = MODELS_DIR / "lstm_metadata_full.json"

# Aynı hyperparameters
SEQUENCE_LEN = 12
LATENT_DIM = 4
HIDDEN_DIM = 12
EPOCHS = 200
BATCH_SIZE = 16
PATIENCE = 20
LEARNING_RATE = 5e-4
KERAS_VAL_SPLIT = 0.15
RANDOM_STATE = 42

tf.keras.utils.set_random_seed(RANDOM_STATE)


def fit_lstm_autoencoder_full(train_df: pd.DataFrame, val_df: pd.DataFrame):
    # !!! TÜRKİYE FİLTRESİ YOK — 13 şehir tamamı dahil
    print(f"[LSTM-FULL] 13 şehir tamamıyla eğitim (Türkiye dahil).")
    print(f"[LSTM-FULL]   Train: {len(train_df)}, val: {len(val_df)}")

    feature_cols = CORE_FEATURES
    n_features = len(feature_cols)

    scaler = StandardScaler()
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols].values)
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols].values)

    print(f"[LSTM-FULL] Sequence oluşturuluyor (seq_len={SEQUENCE_LEN})...")
    X_train, _ = make_sequences(train_scaled, feature_cols, SEQUENCE_LEN)
    X_val, _ = make_sequences(val_scaled, feature_cols, SEQUENCE_LEN)
    print(f"[LSTM-FULL]   Train sequences: {X_train.shape}")
    print(f"[LSTM-FULL]   Val sequences:   {X_val.shape}")

    if X_train.shape[0] < 8:
        print(f"[LSTM-FULL] !!! Yetersiz veri.")
        return None, None, None

    model = build_lstm_autoencoder(SEQUENCE_LEN, n_features)
    model.summary()

    print("[LSTM-FULL] Eğitim başlıyor...")
    if X_val.shape[0] > 0:
        callbacks = [EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)]
        history = model.fit(
            X_train, X_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, X_val),
            callbacks=callbacks,
            verbose=2,
        )
    else:
        callbacks = [EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)]
        history = model.fit(
            X_train, X_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=KERAS_VAL_SPLIT,
            shuffle=True,
            callbacks=callbacks,
            verbose=2,
        )

    train_recon = model.predict(X_train, verbose=0)
    train_mse = np.mean(np.square(X_train - train_recon), axis=(1, 2))
    threshold_p99 = float(np.percentile(train_mse, 99))
    threshold_p95 = float(np.percentile(train_mse, 95))

    val_recon = model.predict(X_val, verbose=0) if X_val.shape[0] > 0 else None
    val_mse = (
        np.mean(np.square(X_val - val_recon), axis=(1, 2)) if val_recon is not None else np.array([])
    )

    print(f"\n[LSTM-FULL] Train MSE — mean={train_mse.mean():.4f}, p95={threshold_p95:.4f}, p99={threshold_p99:.4f}")
    if val_mse.size > 0:
        print(f"[LSTM-FULL] Val   MSE — mean={val_mse.mean():.4f}, max={val_mse.max():.4f}")
        n_val_anomaly = int((val_mse > threshold_p99).sum())
        print(f"[LSTM-FULL] Val anomali sayısı (eşik p99): {n_val_anomaly}/{len(val_mse)}")

    model.save(LSTM_MODEL_PATH)
    joblib.dump(scaler, LSTM_SCALER_PATH)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "version": "FULL (13 cities including Turkey)",
        "feature_columns": feature_cols,
        "sequence_length": SEQUENCE_LEN,
        "latent_dim": LATENT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "n_train_sequences": int(X_train.shape[0]),
        "n_val_sequences": int(X_val.shape[0]),
        "history_train_loss": [float(v) for v in history.history["loss"]],
        "history_val_loss": [float(v) for v in history.history.get("val_loss", [])],
        "history_n_epochs": len(history.history["loss"]),
        "epochs_run": len(history.history["loss"]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history.get("val_loss", [np.nan])[-1]),
        "threshold_p95": threshold_p95,
        "threshold_p99": threshold_p99,
        "train_mse_mean": float(train_mse.mean()),
        "train_mse_std": float(train_mse.std()),
    }
    with open(LSTM_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[LSTM-FULL] Model kaydedildi: {LSTM_MODEL_PATH}")
    print(f"[LSTM-FULL] Scaler kaydedildi: {LSTM_SCALER_PATH}")
    print(f"[LSTM-FULL] Metadata kaydedildi: {LSTM_META_PATH}")

    return model, scaler, metadata


def main():
    print("=" * 60)
    print("LSTM AUTOENCODER TRAINING — FULL (13 cities)")
    print("=" * 60)
    train, val, _ = prepare_dataset()
    fit_lstm_autoencoder_full(train, val)


if __name__ == "__main__":
    main()