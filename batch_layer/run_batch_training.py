"""
Batch Layer — Tek komutla tüm batch eğitimini çalıştırır.

Çalıştırma:
  python -m batch_layer.run_batch_training
"""
from batch_layer.data_prep import prepare_dataset
from batch_layer.train_isolation_forest import fit_isolation_forest
from batch_layer.train_lstm_ae import fit_lstm_autoencoder


def main():
    print("\n" + "=" * 60)
    print("BATCH LAYER TRAINING — IF + LSTM AE")
    print("=" * 60)

    train_df, val_df, _ = prepare_dataset()

    print("\n" + "─" * 60)
    print("Stage 1/2: Isolation Forest")
    print("─" * 60)
    fit_isolation_forest(train_df, val_df)

    print("\n" + "─" * 60)
    print("Stage 2/2: LSTM Autoencoder")
    print("─" * 60)
    fit_lstm_autoencoder(train_df, val_df)

    print("\n" + "=" * 60)
    print("BATCH TRAINING TAMAMLANDI ✓")
    print("=" * 60)
    print("Modeller: data/models/")
    print("  - isolation_forest.joblib + if_scaler.joblib + if_metadata.json")
    print("  - lstm_autoencoder.keras  + lstm_scaler.joblib + lstm_metadata.json")


if __name__ == "__main__":
    main()