# Real-Time Air Quality Anomaly Detection on a Lambda Architecture

Reference implementation for the paper *A Big Data Pipeline Based on the
Lambda Architecture for Real-Time Anomaly Detection in Multivariate Air
Quality Data Streams* (Özyürek & Odabaş Yıldırım).

The pipeline ingests multi-pollutant measurements from the AQICN API through
Apache Kafka, processes them with Spark Structured Streaming, scores each
observation with two unsupervised detectors in parallel (Isolation Forest for
point outliers, an LSTM Autoencoder for temporal deviations), and surfaces the
results on a Streamlit operator dashboard. A rule-based data quality layer
separates hardware saturation from genuine pollution episodes.

## Reproducing the paper

Every dataset-level figure reported in the paper can be regenerated from the
frozen snapshot in `data/snapshot/`:

```bash
python -m tools.verify_paper_numbers
```

This checks 17 reported quantities — ingestion counts, filtering outcomes,
sequence counts, saturation statistics and both decision thresholds — against
values recomputed from the data. All 17 pass on the released snapshot.

## Dataset snapshot

`data/snapshot/aqicn_events_snapshot.parquet` contains the 20,225 raw
measurements collected between 26 April and 13 May 2026 from 39 stations
across 13 cities. All results in the paper derive from this frozen file; the
live pipeline has not been rerun since.

Data originate from the World Air Quality Index project (https://aqicn.org).
The snapshot is redistributed for reproducibility; users intending to collect
new data must obtain their own API token.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # then add your AQICN token
```

Live ingestion additionally requires Kafka 3.7.0 and Spark 3.5.1. Offline
reproduction of the paper's numbers does not.

## Experiment scripts

| Script | Produces |
|---|---|
| `tools/verify_paper_numbers.py` | All dataset-level figures |
| `tools/volume_control.py` | Volume-controlled ablation (Table 12, Figure 3) |
| `tools/drift_replay.py` | Replay-based drift probe (Table 9) |
| `tools/make_figure2.py` | LSTM training curves (Figure 2) |
| `batch_layer/synthetic_injection_test.py` | Injection metrics (Table 19) |

## Pipeline components

| Path | Role |
|---|---|
| `producer/` | AQICN polling, Kafka producer |
| `speed_layer/` | Spark Structured Streaming, mini-batch scorer |
| `batch_layer/` | Data preparation, IF and LSTM-AE training |
| `dashboard/` | Streamlit operator interface |

## License

MIT. See `LICENSE`.
