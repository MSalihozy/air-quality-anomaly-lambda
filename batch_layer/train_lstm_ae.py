"""
Batch Layer — LSTM Autoencoder eğitimi.

Wide formattaki istasyon × saat verisinden, her istasyon için kayan pencere
(sliding window) sequence'leri çıkarır. Encoder→bottleneck→Decoder mimarisinde
LSTM AE eğitir. Reconstruction error'un p99'unu anomali eşiği olarak alır.

Çalıştırma:
  python -m batch_layer.train_lstm_ae
"""
import os
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

# TensorFlow/Keras CPU mode için verbose'u kıs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

# Pylance'in lazy loader sorununu bypass etmek için doğrudan tf.keras üzerinden alıyoruz
Model = tf.keras.Model
Input = tf.keras.layers.Input
LSTM = tf.keras.layers.LSTM
RepeatVector = tf.keras.layers.RepeatVector
TimeDistributed = tf.keras.layers.TimeDistributed
Dense = tf.keras.layers.Dense
EarlyStopping = tf.keras.callbacks.EarlyStopping
from sklearn.preprocessing import StandardScaler

from batch_layer.data_prep import prepare_dataset, CORE_FEATURES

MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LSTM_MODEL_PATH = MODELS_DIR / "lstm_autoencoder.keras"
LSTM_SCALER_PATH = MODELS_DIR / "lstm_scaler.joblib"
LSTM_META_PATH = MODELS_DIR / "lstm_metadata.json"

# Hiperparametreler
SEQUENCE_LEN = 12
LATENT_DIM = 4
HIDDEN_DIM = 12
EPOCHS = 200            # 50 → 200, küçük veri için daha çok iterasyon
BATCH_SIZE = 16         # 32 → 16, küçük data için daha çok güncelleme
PATIENCE = 20           # 8 → 20, gürültülü loss için tolerans
LEARNING_RATE = 5e-4    # 1e-3 → 5e-4, daha stabil iniş
KERAS_VAL_SPLIT = 0.15  # YENİ: Keras kendi içinde val ayırsın (val sequence'ımız 0)
RANDOM_STATE = 42

tf.keras.utils.set_random_seed(RANDOM_STATE)


def make_sequences(df_wide: pd.DataFrame, feature_cols, seq_len: int):
    """
    Her istasyon için zaman-sıralı satırları al, kayan pencere (sliding window)
    sequence'leri çıkar. Returns:
        X: (n_samples, seq_len, n_features) numpy array
        meta: pandas DataFrame [station_id, end_time]
    """
    sequences = []
    metas = []

    for station_id, grp in df_wide.sort_values(["station_id", "event_time"]).groupby("station_id"):
        values = grp[feature_cols].values  # (T, F)
        times = grp["event_time"].values
        if len(values) < seq_len:
            continue
        for i in range(len(values) - seq_len + 1):
            sequences.append(values[i : i + seq_len])
            metas.append({
                "station_id": station_id,
                "end_time": times[i + seq_len - 1],
            })

    if not sequences:
        return np.empty((0, seq_len, len(feature_cols))), pd.DataFrame(metas)

    return np.array(sequences), pd.DataFrame(metas)


def build_lstm_autoencoder(seq_len: int, n_features: int) -> Model:
    """Encoder → bottleneck → Decoder LSTM AE."""
    inputs = Input(shape=(seq_len, n_features))

    # Encoder
    x = LSTM(HIDDEN_DIM, activation="tanh", return_sequences=True)(inputs)
    encoded = LSTM(LATENT_DIM, activation="tanh", return_sequences=False)(x)

    # Bottleneck → Decoder
    x = RepeatVector(seq_len)(encoded)
    x = LSTM(LATENT_DIM, activation="tanh", return_sequences=True)(x)
    x = LSTM(HIDDEN_DIM, activation="tanh", return_sequences=True)(x)
    decoded = TimeDistributed(Dense(n_features))(x)

    model = Model(inputs=inputs, outputs=decoded, name="lstm_autoencoder")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )
    return model


def fit_lstm_autoencoder(train_df: pd.DataFrame, val_df: pd.DataFrame):
    

    # Türkiye istasyonları için yeterli saat dilimi yok — LSTM eğitiminden hariç tut
    # (raporda "AQICN Türkiye verisi seyrek güncellenir" dipnotu)
    TURKEY_CITIES = {"Istanbul", "Ankara", "Antalya", "Erzurum", "Konya", "Bursa", "Canakkale"}

    train_df = train_df[~train_df["city"].isin(TURKEY_CITIES)].reset_index(drop=True)
    val_df = val_df[~val_df["city"].isin(TURKEY_CITIES)].reset_index(drop=True)
    print(f"[LSTM] Türkiye istasyonları hariç tutuldu (yetersiz saat dilimi).")
    print(f"[LSTM]   Filtre sonrası train: {len(train_df)}, val: {len(val_df)}")

    feature_cols = CORE_FEATURES  # LSTM'de zaman-bağlı feature'lar (hour_of_day) yok, sadece core
    n_features = len(feature_cols)

    
    feature_cols = CORE_FEATURES  # LSTM'de zaman-bağlı feature'lar (hour_of_day) yok, sadece core
    n_features = len(feature_cols)

    # 1) Scaling — istasyondan bağımsız, global standardization
    scaler = StandardScaler()
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols].values)
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols].values)

    # 2) Sequence oluştur
    print(f"[LSTM] Sequence oluşturuluyor (seq_len={SEQUENCE_LEN})...")
    X_train, meta_train = make_sequences(train_scaled, feature_cols, SEQUENCE_LEN)
    X_val,   meta_val   = make_sequences(val_scaled,   feature_cols, SEQUENCE_LEN)
    print(f"[LSTM]   Train sequences: {X_train.shape}")
    print(f"[LSTM]   Val sequences:   {X_val.shape}")

    if X_train.shape[0] < 8:
        print(f"[LSTM] !!! Yetersiz veri: {X_train.shape[0]} sequence. "
              f"Daha fazla saat biriktirin (her istasyon en az {SEQUENCE_LEN} saatlik veri vermeli).")
        return None, None, None

    # 3) Model
    model = build_lstm_autoencoder(SEQUENCE_LEN, n_features)
    model.summary()

    # 4) Eğitim — küçük veri durumu için validation split Keras'a bırakıldı
    print("[LSTM] Eğitim başlıyor...")
    if X_val.shape[0] > 0:
        # Time-based split val mevcut, onu kullan
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
        # Time-based val boş — train'den %15 ayırarak Keras'a iç val yaptır
        callbacks = [EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)]
        history = model.fit(
            X_train, X_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=KERAS_VAL_SPLIT,
            shuffle=True,  # küçük data için OK; sequence'lar zaten kendi içinde temporal
            callbacks=callbacks,
            verbose=2,
        )

    # 5) Reconstruction error → eşik (p99 of train residuals)
    train_recon = model.predict(X_train, verbose=0)
    train_mse = np.mean(np.square(X_train - train_recon), axis=(1, 2))
    threshold_p99 = float(np.percentile(train_mse, 99))
    threshold_p95 = float(np.percentile(train_mse, 95))

    val_recon = model.predict(X_val, verbose=0) if X_val.shape[0] > 0 else None
    val_mse = (
        np.mean(np.square(X_val - val_recon), axis=(1, 2)) if val_recon is not None else np.array([])
    )

    print(f"\n[LSTM] Train MSE — mean={train_mse.mean():.4f}, p95={threshold_p95:.4f}, p99={threshold_p99:.4f}")
    if val_mse.size > 0:
        print(f"[LSTM] Val   MSE — mean={val_mse.mean():.4f}, max={val_mse.max():.4f}")
        n_val_anomaly = int((val_mse > threshold_p99).sum())
        print(f"[LSTM] Val anomali sayısı (eşik p99): {n_val_anomaly}/{len(val_mse)}")

    # 6) Kaydet
    model.save(LSTM_MODEL_PATH)
    joblib.dump(scaler, LSTM_SCALER_PATH)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
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

    print(f"\n[LSTM] Model kaydedildi: {LSTM_MODEL_PATH}")
    print(f"[LSTM] Scaler kaydedildi: {LSTM_SCALER_PATH}")
    print(f"[LSTM] Metadata kaydedildi: {LSTM_META_PATH}")

    return model, scaler, metadata


def main():
    print("=" * 60)
    print("LSTM AUTOENCODER TRAINING")
    print("=" * 60)
    train, val, _ = prepare_dataset()
    fit_lstm_autoencoder(train, val)


if __name__ == "__main__":
    main()