"""
OpenAQ Air Quality Anomaly Dashboard
Streamlit + Plotly + OpenStreetMap

Veri kaynakları:
  - data/raw_parquet/events/   → speed layer raw events
  - data/raw_parquet/scored/   → mini-batch scorer IF skorlu
  - data/models/if_metadata.json → IF model metadata

Çalıştırma:
  streamlit run dashboard/app.py
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Yapılandırma ---
EVENTS_PATH = Path("data/raw_parquet/events")
SCORED_PATH = Path("data/raw_parquet/scored")
IF_META_PATH = Path("data/models/if_metadata.json")

# Streamlit config
st.set_page_config(
    page_title="Air Quality Anomaly Detection",
    page_icon="🌫️",
    layout="wide",
)

# --- Cache'lenmiş veri yükleyiciler ---
@st.cache_data(ttl=30)
def load_events():
    """Raw events Parquet → DataFrame."""
    if not EVENTS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(EVENTS_PATH)
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    return df


@st.cache_data(ttl=30)
def load_scored():
    """Skorlanmış (IF) Parquet → DataFrame."""
    if not SCORED_PATH.exists() or not any(SCORED_PATH.iterdir()):
        return pd.DataFrame()
    df = pd.read_parquet(SCORED_PATH)
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    df["scored_at"] = pd.to_datetime(df["scored_at"], utc=True)
    return df


@st.cache_data(ttl=300)
def load_if_metadata():
    """IF metadata (eğitim zamanı, sample sayısı vb.)."""
    if not IF_META_PATH.exists():
        return {}
    with open(IF_META_PATH) as f:
        return json.load(f)


# --- Veri yükle ---
events = load_events()
scored = load_scored()
if_meta = load_if_metadata()

# --- Sidebar: filtreler + bilgi ---
with st.sidebar:
    # Manuel refresh
    col_r1, col_r2 = st.columns([1, 1])
    with col_r1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col_r2:
        auto_refresh = st.toggle("Auto", value=False, help="Auto-refresh every 30 second")

    if auto_refresh:
        st.caption("⏱️ Refreshing every 30s...")

    st.divider()
    st.header("⚙️ Filters")

    if events.empty:
        st.warning("No data yet. Producer + Spark must be running.")
        st.stop()

    cities = sorted(events["city"].unique().tolist())
    selected_cities = st.multiselect("Cities", cities, default=cities)

    time_window_hours = st.slider(
        "Time window (last N hours)",
        min_value=6, max_value=360, value=240, step=12,
    )

    st.divider()
    st.header("🧠 Model Info")
    if if_meta:
        trained = if_meta.get("trained_at", "—")
        n_train = if_meta.get("n_train_samples", "—")
        contam = if_meta.get("hyperparameters", {}).get("contamination", "—")
        st.caption(f"**Isolation Forest**")
        st.text(f"Trained: {trained[:19] if trained != '—' else '—'}")
        st.text(f"Train sample: {n_train}")
        st.text(f"Contamination: {contam}")

    st.divider()
    st.caption(
        "**Data sources:**\n\n"
        "- AQICN API (Turkey, Europe, India, USA)\n\n"
        "**Pipeline:** Kafka → Spark Structured Streaming → Parquet → IF Scoring"
    )

# --- Filtreleri uygula ---
now = datetime.now(timezone.utc)
cutoff = now - timedelta(hours=time_window_hours)

events_filt = events[
    (events["city"].isin(selected_cities)) &
    (events["event_time"] >= cutoff)
].copy()

scored_filt = scored.copy() if not scored.empty else pd.DataFrame()
if not scored_filt.empty:
    scored_filt = scored_filt[
        (scored_filt["city"].isin(selected_cities)) &
        (scored_filt["event_time"] >= cutoff)
    ]

# --- Header ---
st.title("🌫️ Global Air Quality Anomaly Detection")
st.caption(
    f"Lambda Architecture · Apache Kafka + Spark Structured Streaming · Isolation Forest "
    f"| Last update: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}"
)

# --- KPI Band ---
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    n_stations = events_filt["station_id"].nunique() if not events_filt.empty else 0
    st.metric("Active stations", n_stations)

with col2:
    n_events = len(events_filt)
    st.metric("Measurements", f"{n_events:,}")

with col3:
    n_if = int(scored_filt["is_anomaly_if"].sum()) if not scored_filt.empty and "is_anomaly_if" in scored_filt.columns else 0
    st.metric("IF anomalies", n_if)

with col4:
    n_lstm = int(scored_filt["is_anomaly_lstm"].fillna(False).sum()) if not scored_filt.empty and "is_anomaly_lstm" in scored_filt.columns else 0
    st.metric("LSTM anomalies", n_lstm)

with col5:
    pm25_data = events_filt[events_filt["parameter"] == "pm25"]["value"]
    avg_pm25 = pm25_data.mean() if not pm25_data.empty else 0
    if avg_pm25 <= 15:
        delta_text = "Below WHO ✓"
        delta_color = "normal"
    elif avg_pm25 <= 35:
        delta_text = "Above WHO"
        delta_color = "off"
    else:
        delta_text = "High ⚠️"
        delta_color = "inverse"
    st.metric("Average PM2.5 (µg/m³)", f"{avg_pm25:.1f}", delta=delta_text, delta_color=delta_color)

st.divider()

# --- Sol: Harita | Sağ: Trend grafiği ---
col_map, col_trend = st.columns([3, 2])

with col_map:
    st.subheader("📍 Station map")

    if scored_filt.empty:
        st.info("No scored data. Run python -m speed_layer.mini_batch_scorer.")
    else:
        agg = (
            scored_filt.groupby(["station_id", "city", "station_name", "lat", "lon"], observed=True)
            .agg(
                anomaly_count=("is_anomaly", "sum"),
                total_count=("is_anomaly", "count"),
                last_pm25=("pm25", "last"),
                min_score=("if_score", "min"),
            )
            .reset_index()
        )
        agg["anomaly_rate"] = (agg["anomaly_count"] / agg["total_count"] * 100).round(1)
        agg["status"] = agg["anomaly_count"].apply(
            lambda x: "Anomalous" if x > 0 else "Normal"
        )

        fig_map = px.scatter_mapbox(
            agg,
            lat="lat",
            lon="lon",
            color="status",
            size="last_pm25",
            hover_name="station_name",
            hover_data={
                "city": True,
                "anomaly_count": True,
                "total_count": True,
                "anomaly_rate": ":.1f",
                "last_pm25": ":.1f",
                "min_score": ":.3f",
                "lat": False,
                "lon": False,
            },
            color_discrete_map={"Anomalous": "#e74c3c", "Normal": "#2ecc71"},
            size_max=30,
            zoom=1,
            center={"lat": 30, "lon": 30},  # YENİ: harita merkezi
            height=550,
        )
        fig_map.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_map, use_container_width=True)

with col_trend:
    st.subheader("📈 Station trend")

    if events_filt.empty:
        st.info("No data to display.")
    else:
        station_options = (
            events_filt[["station_id", "station_name", "city"]]
            .drop_duplicates()
            .sort_values(["city", "station_name"])
        )
        station_labels = station_options.apply(
            lambda r: f"{r['city']} — {r['station_name'] or 'unnamed'}", axis=1
        ).tolist()
        station_ids = station_options["station_id"].tolist()

        default_idx = 0
        if not scored_filt.empty:
            top_anom_station = (
                scored_filt[scored_filt["is_anomaly"]]
                .groupby("station_id", observed=True).size()
                .sort_values(ascending=False)
            )
            if len(top_anom_station) > 0 and top_anom_station.index[0] in station_ids:
                default_idx = station_ids.index(top_anom_station.index[0])

        selected_label = st.selectbox(
            "Select station",
            options=station_labels,
            index=default_idx,
        )
        selected_station_id = station_ids[station_labels.index(selected_label)]

        st_data = events_filt[events_filt["station_id"] == selected_station_id]
        st_data = st_data[st_data["parameter"].isin(["pm25", "pm10", "no2", "co"])]

        st_anom = pd.DataFrame()
        if not scored_filt.empty:
            st_anom = scored_filt[
                (scored_filt["station_id"] == selected_station_id) &
                (scored_filt["is_anomaly"])
            ]

        if st_data.empty:
            st.info("No parameters to display for this station.")
        elif st_data["event_time"].nunique() < 3:
            n_points = st_data["event_time"].nunique()
            st.info(
                f"📊 Only **{n_points} time slots** of data have arrived from this station so far. "
                "At least 3 time slots are needed for a meaningful trend chart. "
                "This chart will become richer as the producer keeps running."
            )
            # Yine de noktaları göster (bilgi olarak)
            st.dataframe(
                st_data[["event_time", "parameter", "value"]].sort_values(["parameter", "event_time"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            # Sensör doyumu eşikleri (data quality)
            SAT_LIMITS = {"pm25": 500.0, "pm10": 999.0}

            df_sorted = st_data.sort_values("event_time").copy()

            # Her ölçümü "saturated" olarak etiketle
            df_sorted["is_saturated"] = df_sorted.apply(
                lambda r: (r["parameter"] in SAT_LIMITS) and (r["value"] >= SAT_LIMITS[r["parameter"]]),
                axis=1,
            )

            df_clean = df_sorted[~df_sorted["is_saturated"]]
            df_sat = df_sorted[df_sorted["is_saturated"]]

            # Ana trend grafiği — sadece temiz değerler
            if not df_clean.empty:
                fig_trend = px.line(
                    df_clean,
                    x="event_time",
                    y="value",
                    color="parameter",
                    markers=True,
                    labels={"value": "Value (µg/m³)", "event_time": "Time"},
                    height=400,
                )
            else:
                # Hiç temiz değer yok — boş grafik başlat
                fig_trend = go.Figure()
                fig_trend.update_layout(
                    height=400,
                    xaxis_title="Time",
                    yaxis_title="Value (µg/m³)",
                )

            # Sensör doyumu noktalarını gri X olarak ekle
            if not df_sat.empty:
                fig_trend.add_trace(
                    go.Scatter(
                        x=df_sat["event_time"],
                        y=df_sat["value"],
                        mode="markers",
                        marker=dict(symbol="x", size=12, color="#888888", line=dict(color="#444444", width=1)),
                        name="Sensor saturation (data quality)",
                        hovertemplate="<b>SENSOR SATURATION</b><br>Time: %{x}<br>%{text}: %{y}<extra></extra>",
                        text=df_sat["parameter"],
                    )
                )

            # Kırmızı yıldız anomali noktaları (PM2.5)
            if not st_anom.empty and "pm25" in st_anom.columns:
                # Sadece eşik altındaki gerçek anomali yıldızlarını göster
                real_anom = st_anom[st_anom["pm25"] < SAT_LIMITS["pm25"]]
                if not real_anom.empty:
                    fig_trend.add_trace(
                        go.Scatter(
                            x=real_anom["event_time"],
                            y=real_anom["pm25"],
                            mode="markers",
                            marker=dict(symbol="star", size=18, color="red", line=dict(color="black", width=1)),
                            name="Anomaly (PM2.5)",
                            hovertemplate="<b>ANOMALY</b><br>Time: %{x}<br>PM2.5: %{y}<extra></extra>",
                        )
                    )

            fig_trend.update_layout(
                margin={"r": 10, "t": 10, "l": 10, "b": 10},
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )

            # Sensör doyumu varsa kullanıcıya açıklama göster
            if not df_sat.empty:
                n_sat = len(df_sat)
                st.caption(
                    f"⚠️ **Data quality note:** At this station, {n_sat} measurements are stuck at the sensor saturation limit "
                    f"(PM2.5 ≥ {SAT_LIMITS['pm25']:.0f} or PM10 ≥ {SAT_LIMITS['pm10']:.0f}). "
                    f"These are shown with a gray X marker, separate from genuine anomalies."
                )

            st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# --- Alt blok: Şehir başına anomali + Son anomaliler tablosu ---
col_bar, col_table = st.columns([1, 2])

with col_bar:
    st.subheader("🔬 IF vs LSTM AE comparison")
    if scored_filt.empty or "is_anomaly_lstm" not in scored_filt.columns:
        st.info("No scored data or LSTM column is missing.")
    else:
        # 3 kategori: sadece IF, sadece LSTM, ortak
        only_if = ((scored_filt["is_anomaly_if"]) & (~scored_filt["is_anomaly_lstm"].fillna(False))).sum()
        only_lstm = ((~scored_filt["is_anomaly_if"]) & (scored_filt["is_anomaly_lstm"].fillna(False))).sum()
        both = ((scored_filt["is_anomaly_if"]) & (scored_filt["is_anomaly_lstm"].fillna(False))).sum()

        comparison_df = pd.DataFrame({
            "Kategori": ["IF only", "LSTM AE only", "Both (IF ∩ LSTM)"],
            "Sayı": [int(only_if), int(only_lstm), int(both)],
        })

        fig_bar = px.bar(
            comparison_df,
            x="Kategori", y="Sayı",
            color="Kategori",
            color_discrete_map={
                "IF only": "#3498db",
                "LSTM AE only": "#9b59b6",
                "Both (IF ∩ LSTM)": "#e74c3c",
            },
            text="Sayı",
            height=400,
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            margin={"r": 10, "t": 10, "l": 10, "b": 10},
            showlegend=False,
            xaxis_title="",
            yaxis_title="Anomaly count",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.caption(
            f"💡 **Both** ({int(both)} anomalies) — confirmed by two different paradigms, highest confidence. "
            f"**LSTM only** ({int(only_lstm)}) — temporal pattern disruption. "
            f"**IF only** ({int(only_if)}) — point outlier."
        )

with col_table:
    st.subheader("🚨 Recent anomalies")
    if scored_filt.empty:
        st.info("No anomalies yet.")
    else:
        # Tespit eden modeli etiketle
        sf = scored_filt[scored_filt["is_anomaly"]].copy()
        if "is_anomaly_lstm" in sf.columns:
            sf["lstm_safe"] = sf["is_anomaly_lstm"].fillna(False)
            sf["Detected by"] = sf.apply(
                lambda r: "🔴 Both" if r["is_anomaly_if"] and r["lstm_safe"]
                else ("🟣 LSTM" if r["lstm_safe"]
                else "🔵 IF"), axis=1
            )
        else:
            sf["Detected by"] = "🔵 IF"

        cols_to_show = ["event_time", "Detected by", "city", "station_name", "pm25", "co", "no2", "if_score"]
        if "lstm_mse" in sf.columns:
            cols_to_show.append("lstm_mse")

        anom_table = (
            sf.nsmallest(20, "if_score")
            [cols_to_show]
            .rename(columns={
                "event_time": "Time",
                "city": "City",
                "station_name": "Station",
                "pm25": "PM2.5",
                "co": "CO",
                "no2": "NO2",
                "if_score": "IF Score",
                "lstm_mse": "LSTM MSE",
            })
        )
        anom_table["Time"] = anom_table["Time"].dt.strftime("%Y-%m-%d %H:%M UTC")
        st.dataframe(
            anom_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "IF Score": st.column_config.NumberColumn(format="%.4f"),
                "LSTM MSE": st.column_config.NumberColumn(format="%.3f"),
                "PM2.5": st.column_config.NumberColumn(format="%.0f"),
                "CO": st.column_config.NumberColumn(format="%.1f"),
                "NO2": st.column_config.NumberColumn(format="%.1f"),
            },
        )

# --- Footer ---
st.divider()
st.caption(
    "© 2026 Mehmet Salih Özyürek — Atatürk University · Master's Project · "
    "Apache Kafka + Spark Structured Streaming + Lambda Architecture + Isolation Forest"
)

# --- Auto-refresh (eğer toggle açıksa) ---
if auto_refresh:
    import time
    time.sleep(30)
    st.cache_data.clear()
    st.rerun()