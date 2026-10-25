"""
Batch Layer — Veri hazırlama modülü.

Speed layer'ın yazdığı raw Parquet event'lerini okur, long→wide pivot eder,
feature engineering uygular, train/val split döndürür.

Diğer modüller (IF, LSTM AE eğitimi) bu modülü import eder.
"""
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# --- Yapılandırma ---
RAW_EVENTS_PATH = "data/raw_parquet/events"

# 6 kirlilik + 4 meteoroloji = 10 ana feature
POLLUTION_FEATURES = ["pm25", "pm10", "co", "no2", "so2", "o3"]
METEO_FEATURES = ["t", "h", "p", "w"]
CORE_FEATURES = POLLUTION_FEATURES + METEO_FEATURES

# Pivot sonrası eklenen mühendislik feature'ları
ENGINEERED_FEATURES = ["hour_of_day", "day_of_week"]

ALL_FEATURES = CORE_FEATURES + ENGINEERED_FEATURES


def load_raw_events(path: str = RAW_EVENTS_PATH) -> pd.DataFrame:
    """Spark'ın yazdığı partition'lı Parquet'i tek DataFrame'e oku."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Raw events Parquet bulunamadı: {path}")
    df = pd.read_parquet(path)
    # event_time'ı timezone-aware datetime'a çevir
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    return df


def long_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Long formatı (her satır bir parametre ölçümü) wide'a çevir.
    Bir satır = bir istasyon × bir saat, sütunlar = parametreler.
    """
    # Tekil ölçümleri al — aynı (station, time, parameter) için son değeri tut
    df_long = df_long.sort_values("ingested_time").drop_duplicates(
        subset=["station_id", "event_time", "parameter"], keep="last"
    )

    wide = df_long.pivot_table(
        index=["station_id", "city", "station_name", "lat", "lon", "event_time"],
        columns="parameter",
        values="value",
        aggfunc="last",
        observed=True,
    ).reset_index()

    # Pivot sonrası kolon adı index'in adını korur — temizle
    wide.columns.name = None

    # CORE_FEATURES'ta olup wide'da olmayan parametreler için NaN sütun ekle
    for f in CORE_FEATURES:
        if f not in wide.columns:
            wide[f] = np.nan

    return wide


def add_engineered_features(wide: pd.DataFrame) -> pd.DataFrame:
    """Zaman bazlı feature ekle: hour_of_day, day_of_week."""
    out = wide.copy()
    out["hour_of_day"] = out["event_time"].dt.hour
    out["day_of_week"] = out["event_time"].dt.dayofweek
    return out


def filter_complete_rows(wide: pd.DataFrame, min_features: int = 8) -> pd.DataFrame:
    """
    Bir satırda en az `min_features` kadar feature dolu olmalı, yoksa at.
    LA/Houston'da pm1, dew vb. eksik olabiliyor — bu filtre dengeyi tutuyor.
    """
    feat_cols = CORE_FEATURES
    valid_count = wide[feat_cols].notna().sum(axis=1)
    return wide[valid_count >= min_features].copy()


def impute_missing(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Kalan NaN'ları doldur. Strategi: istasyon başına forward fill,
    sonra kalanları o şehir'in medyanı ile.
    """
    out = wide.copy().sort_values(["station_id", "event_time"])

    # 1) İstasyon başına forward fill
    out[CORE_FEATURES] = (
        out.groupby("station_id")[CORE_FEATURES]
           .ffill()
           .bfill()
    )

    # 2) Hâlâ NaN varsa şehir medyanı
    for col in CORE_FEATURES:
        if out[col].isna().any():
            city_median = out.groupby("city")[col].transform("median")
            out[col] = out[col].fillna(city_median)
        # En kötü ihtimal: global medyan
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())

    return out


def time_based_split(
    wide: pd.DataFrame,
    val_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based train/val split: zaman serisi olduğu için son %20 validation,
    önceki %80 train. Random split yapmıyoruz çünkü leak olur.
    """
    sorted_df = wide.sort_values("event_time").reset_index(drop=True)
    cutoff = int(len(sorted_df) * (1 - val_fraction))
    train = sorted_df.iloc[:cutoff].copy()
    val = sorted_df.iloc[cutoff:].copy()
    return train, val


def prepare_dataset(min_features: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Uçtan uca pipeline: parquet'i oku, wide'a çevir, FE, impute, split.
    Returns (train_df, val_df, full_wide_df) — tüm DF'ler ALL_FEATURES içerir.
    """
    print("[data_prep] Raw events okunuyor...")
    df = load_raw_events()
    print(f"[data_prep]   Long format: {len(df)} satır, {df['station_id'].nunique()} istasyon")

    print("[data_prep] Long → wide pivot...")
    wide = long_to_wide(df)
    print(f"[data_prep]   Wide format: {len(wide)} satır × {len(wide.columns)} kolon")

    print("[data_prep] Engineered feature'lar ekleniyor...")
    wide = add_engineered_features(wide)

    print(f"[data_prep] Eksik feature filtresi (min {min_features} core feature)...")
    before = len(wide)
    wide = filter_complete_rows(wide, min_features=min_features)
    print(f"[data_prep]   {before} → {len(wide)} satır")

    print("[data_prep] Missing imputation...")
    wide = impute_missing(wide)

    print("[data_prep] Train/val split...")
    train, val = time_based_split(wide, val_fraction=0.2)
    print(f"[data_prep]   Train: {len(train)} | Val: {len(val)}")

    return train, val, wide


if __name__ == "__main__":
    # Bu modül direkt çalıştırılırsa diagnostic mode
    train, val, full = prepare_dataset()
    print("\n=== Diagnostics ===")
    print(f"Toplam satır: {len(full)}")
    print(f"Şehir dağılımı:\n{full['city'].value_counts()}")
    print(f"\nFeature istatistikleri (train):")
    print(train[CORE_FEATURES].describe().round(2))
    print(f"\nİstasyon sayısı: {full['station_id'].nunique()}")
    print(f"Zaman aralığı: {full['event_time'].min()} → {full['event_time'].max()}")