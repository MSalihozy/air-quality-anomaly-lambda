import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from batch_layer.data_prep import prepare_dataset, CORE_FEATURES
from batch_layer.train_lstm_ae import make_sequences, build_lstm_autoencoder

TURKEY = {"Istanbul","Ankara","Antalya","Erzurum","Konya","Bursa","Canakkale"}
SEEDS  = [42, 43, 44, 45, 46]
EPOCHS, BATCH, PATIENCE, LR = 200, 16, 20, 5e-4

train, val, _ = prepare_dataset()
val_noTR = val[~val["city"].isin(TURKEY)].reset_index(drop=True)

def egit(train_df, seed, n_alt=None):
    tf.keras.utils.set_random_seed(seed)
    sc = StandardScaler()
    tr = train_df.copy()
    tr[CORE_FEATURES] = sc.fit_transform(train_df[CORE_FEATURES].values)
    vl = val_noTR.copy()
    vl[CORE_FEATURES] = sc.transform(val_noTR[CORE_FEATURES].values)

    Xtr, _ = make_sequences(tr, CORE_FEATURES, 12)
    Xvl, _ = make_sequences(vl, CORE_FEATURES, 12)

    if n_alt is not None and len(Xtr) > n_alt:
        idx = np.random.RandomState(seed).choice(len(Xtr), n_alt, replace=False)
        Xtr = Xtr[idx]

    m = build_lstm_autoencoder(12, len(CORE_FEATURES))
    m.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="mse")
    h = m.fit(Xtr, Xtr, epochs=EPOCHS, batch_size=BATCH,
              validation_data=(Xvl, Xvl), verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor="val_loss", patience=PATIENCE,
                  restore_best_weights=True)])
    return len(Xtr), float(min(h.history["val_loss"]))

train_noTR = train[~train["city"].isin(TURKEY)].reset_index(drop=True)
n_A, _ = egit(train_noTR, 42)
print(f"Version A dizi sayisi: {n_A}\n")

sonuc = {}
for ad, df_, alt in [("A", train_noTR, None), ("B", train, None), ("C", train, n_A)]:
    kayiplar = []
    for s in SEEDS:
        n, v = egit(df_, s, alt)
        kayiplar.append(v)
        print(f"  {ad} seed={s}  n={n:>4}  val={v:.4f}")
    sonuc[ad] = {"n": n, "losses": kayiplar,
                 "mean": float(np.mean(kayiplar)),
                 "std": float(np.std(kayiplar)),
                 "min": float(np.min(kayiplar)),
                 "max": float(np.max(kayiplar))}
    print(f"  --> {ad}: {sonuc[ad]['mean']:.4f} +/- {sonuc[ad]['std']:.4f}\n")

json.dump(sonuc, open("data/volume_control_results.json","w"), indent=2)

a, b, c = sonuc["A"]["mean"], sonuc["B"]["mean"], sonuc["C"]["mean"]
print("="*50)
print(f"A (347 dizi,  7 sehir): {a:.4f}")
print(f"C (347 dizi, 13 sehir): {c:.4f}")
print(f"B (793 dizi, 13 sehir): {b:.4f}")
print(f"\nCesitlilik etkisi (A->C): {(1-c/a)*100:+.1f}%")
print(f"Hacim etkisi     (C->B): {(1-b/c)*100:+.1f}%")
