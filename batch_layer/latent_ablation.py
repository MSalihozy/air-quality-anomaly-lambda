"""
Latent Boyut Ablasyon Çalışması — Bölüm IV-C / VI Doğrulaması

LSTM Autoencoder'ın gizli (latent) boyutunun model performansına etkisini
ölçer. Aynı veri, aynı mimari, aynı eğitim parametreleri kullanılır;
yalnızca latent_dim değişir: {2, 4, 8, 16}.

Bu, "latent_dim neden 4?" sorusuna deneysel cevap üretir.

Çalıştırma:
  cd ~/projects/openaq-anomaly
  source .venv/bin/activate
  python -m batch_layer.latent_ablation

Çıktı:
  - Konsola karşılaştırma tablosu
  - data/latent_ablation_results.json
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

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
from batch_layer.train_lstm_ae import make_sequences

# --- Sabit eğitim parametreleri (train_lstm_ae_full.py ile birebir aynı) ---
SEQUENCE_LEN = 12
HIDDEN_DIM = 12
EPOCHS = 200
BATCH_SIZE = 16
PATIENCE = 20
LEARNING_RATE = 5e-4
RANDOM_STATE = 42

# Ablasyon yapılacak latent boyutları
LATENT_DIMS = [2, 4, 8, 16]


def build_ae_with_latent(seq_len, n_features, latent_dim):
    """train_lstm_ae.py'deki mimariyle birebir aynı, ama latent_dim parametrik."""
    inputs = Input(shape=(seq_len, n_features))
    # Encoder
    x = LSTM(HIDDEN_DIM, activation="tanh", return_sequences=True)(inputs)
    encoded = LSTM(latent_dim, activation="tanh", return_sequences=False)(x)
    # Bottleneck → Decoder
    x = RepeatVector(seq_len)(encoded)
    x = LSTM(latent_dim, activation="tanh", return_sequences=True)(x)
    x = LSTM(HIDDEN_DIM, activation="tanh", return_sequences=True)(x)
    decoded = TimeDistributed(Dense(n_features))(x)
    model = Model(inputs=inputs, outputs=decoded, name=f"lstm_ae_latent{latent_dim}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )
    return model


def run_ablation():
    print("=" * 70)
    print("LATENT BOYUT ABLASYON ÇALIŞMASI")
    print("=" * 70)

    # Veriyi bir kez hazırla (tüm latent boyutları için aynı veri)
    print("\n[1] Veri hazırlanıyor (tüm konfigürasyonlar için ortak)...")
    train_df, val_df, _ = prepare_dataset()
    feature_cols = CORE_FEATURES
    n_features = len(feature_cols)

    # Scaler bir kez fit edilir (veri aynı)
    scaler = StandardScaler()
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols].values)
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols].values)

    X_train, _ = make_sequences(train_scaled, feature_cols, SEQUENCE_LEN)
    X_val, _ = make_sequences(val_scaled, feature_cols, SEQUENCE_LEN)
    print(f"    Train sequences: {X_train.shape}")
    print(f"    Val sequences:   {X_val.shape}")

    if X_train.shape[0] < 8:
        print("    !!! Yetersiz veri. Çıkılıyor.")
        return

    results = []

    # Her latent boyutu için eğit
    for latent_dim in LATENT_DIMS:
        print(f"\n{'='*70}")
        print(f"[2] LATENT_DIM = {latent_dim} eğitiliyor...")
        print(f"{'='*70}")

        # Tekrarlanabilirlik için her modelde aynı tohum
        tf.keras.utils.set_random_seed(RANDOM_STATE)

        model = build_ae_with_latent(SEQUENCE_LEN, n_features, latent_dim)
        n_params = model.count_params()
        print(f"    Toplam parametre: {n_params}")

        callbacks = [EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                    restore_best_weights=True)]
        history = model.fit(
            X_train, X_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, X_val),
            callbacks=callbacks,
            verbose=0,  # sessiz — sadece özet yazdıracağız
        )

        final_train_loss = float(history.history["loss"][-1])
        final_val_loss = float(history.history["val_loss"][-1])
        best_val_loss = float(min(history.history["val_loss"]))
        n_epochs = len(history.history["loss"])

        # Train MSE ve eşik
        train_recon = model.predict(X_train, verbose=0)
        train_mse = np.mean(np.square(X_train - train_recon), axis=(1, 2))
        threshold_p99 = float(np.percentile(train_mse, 99))

        print(f"    → Epoch sayısı:      {n_epochs}")
        print(f"    → Final train loss:  {final_train_loss:.4f}")
        print(f"    → Final val loss:    {final_val_loss:.4f}")
        print(f"    → En iyi val loss:   {best_val_loss:.4f}")
        print(f"    → Eşik p99:          {threshold_p99:.4f}")

        results.append({
            "latent_dim": latent_dim,
            "n_params": int(n_params),
            "n_epochs": n_epochs,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "best_val_loss": best_val_loss,
            "threshold_p99": threshold_p99,
            "train_val_ratio": final_val_loss / final_train_loss if final_train_loss > 0 else 0,
        })

    # ================================================================
    # SONUÇ TABLOSU
    # ================================================================
    print(f"\n\n{'='*70}")
    print("LATENT BOYUT ABLASYON SONUÇLARI")
    print(f"{'='*70}")
    print(f"{'Latent':>7} {'Param':>7} {'Epoch':>6} {'TrainLoss':>10} "
          f"{'ValLoss':>9} {'BestVal':>9} {'Tr/Val':>7}")
    print("-" * 70)
    for r in results:
        marker = "  <-- mevcut" if r["latent_dim"] == 4 else ""
        print(f"{r['latent_dim']:>7} {r['n_params']:>7} {r['n_epochs']:>6} "
              f"{r['final_train_loss']:>10.4f} {r['final_val_loss']:>9.4f} "
              f"{r['best_val_loss']:>9.4f} {r['train_val_ratio']:>7.2f}{marker}")
    print("-" * 70)

    # En iyi latent (en düşük val loss)
    best = min(results, key=lambda r: r["best_val_loss"])
    print(f"\nEn düşük doğrulama kaybı: latent_dim = {best['latent_dim']} "
          f"(best_val_loss = {best['best_val_loss']:.4f})")

    # Kaydet
    output = {
        "ablation_type": "latent_dimension",
        "tested_at": datetime.now(timezone.utc).isoformat(),
        "fixed_params": {
            "sequence_length": SEQUENCE_LEN,
            "hidden_dim": HIDDEN_DIM,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "patience": PATIENCE,
            "random_state": RANDOM_STATE,
        },
        "results": results,
        "best_latent_dim": best["latent_dim"],
    }
    output_path = Path("data/latent_ablation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSonuçlar kaydedildi: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_ablation()