# Implementation Plan: Improved AQI Forecasting Methodology

**Based on:** Literature review of 10 papers (2022–2026) + gap analysis of `AQI_Analysis.ipynb`
**Dataset:** Slovenian Air Quality Dataset — 21 stations, May 2024 – Dec 2025, hourly

---

## Executive Summary

The current notebook (`analysis/AQI_Analysis.ipynb`) has solid data profiling and EDA, but the forecasting phase has a **critical flaw**: the Random Forest model (MAE 3.67) lost to the naive persistence baseline (MAE 3.23) because it was fed no lag features. Without lag features, the ML model is flying blind — it can't see recent trend and has nothing to outperform a "predict tomorrow = today" heuristic.

This plan lays out a **3-phase progressive methodology** that builds from fixing that flaw all the way to a spatially-aware, decomposition-driven deep learning pipeline, grounded in the best techniques from the reviewed papers.

---

## Phase 0: Fix the Existing Notebook (Pre-requisite)

> **Priority: CRITICAL** — Must be done before any new modelling.

### 0.1 Add Lag Features to the Feature Set

**Problem (from notebook):** The Random Forest received only current-time meteorological variables and no memory of recent PM10 values.

**Fix:** Engineer the following lag and rolling features before modelling:

```python
for lag in [1, 2, 3, 6, 12, 24, 48]:
    df_model[f'PM10_lag_{lag}h'] = df_model['PM10'].shift(lag)
    df_model[f'PM25_lag_{lag}h'] = df_model['PM2.5'].shift(lag)

# Rolling statistics
df_model['PM10_rolling_mean_6h']  = df_model['PM10'].rolling(6).mean()
df_model['PM10_rolling_mean_24h'] = df_model['PM10'].rolling(24).mean()
df_model['PM10_rolling_std_24h']  = df_model['PM10'].rolling(24).std()
df_model['PM10_rolling_max_24h']  = df_model['PM10'].rolling(24).max()
```

**Expected outcome:** RF MAE should drop well below the naive baseline (3.23). Papers 2, 10 confirm this as the single highest-impact fix.

### 0.2 Add MAPE as a Third Metric

All 10 reviewed papers report MAPE alongside MAE and RMSE. Add it:

```python
def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

print(f"Naive Baseline MAPE: {mape(y_test, y_pred_naive):.2f}%")
print(f"Random Forest  MAPE: {mape(y_test, y_pred_rf):.2f}%")
```

### 0.3 Evaluate on All 21 Stations (not just E421)

Loop the model across all stations and report per-station and aggregate metrics in a table. This gives a complete, publishable result.

---

## Phase 1: Classical Forecasting Baselines

> **Priority: HIGH** — Establishes a stronger baseline before deep learning.
> **Draws from:** Papers 1, 3, 10 (all benchmark against statistical baselines).

### 1.1 STL Decomposition

Decompose the PM10 time series into Trend + Seasonal + Residual components before modelling. This is the single most impactful preprocessing step identified in the literature (Paper 10: 49% MAPE reduction vs. raw LSTM).

```
PM10(t) = Trend(t) + Seasonal(t) + Residual(t)
```

- Use `statsmodels.tsa.seasonal.STL` with `period=24` (diurnal) and optionally `period=8760` (annual).
- Fit models independently on each component and sum predictions.
- **Implementation file:** `analysis/phase1_stl.py`

### 1.2 SARIMA / SARIMAX Baseline

Replace the naive persistence baseline with a proper seasonal ARIMA model that accounts for the 24-hour diurnal cycle. This is the classical gold standard.

```
SARIMA(p, d, q)(P, D, Q, s=24)
```

- Run on a single station (E421) first as a sanity check.
- Include meteorological covariates (temperature, wind_speed, pressure) as exogenous variables in SARIMAX.
- Evaluate with MAE, RMSE, MAPE on the same 80/20 chronological split.
- **Implementation file:** `analysis/phase1_sarima.py`

### 1.3 Gradient Boosting (XGBoost / LightGBM)

Gradient boosting trees have consistently outperformed vanilla Random Forests on tabular time-series data due to their ability to model nonlinear interactions with less hyperparameter sensitivity.

- Use the full feature set from Phase 0 (lags + rolling + meteorological + time-of-day/month).
- Run `GridSearchCV` (time-series aware) to tune `n_estimators`, `max_depth`, `learning_rate`.
- **Implementation file:** `analysis/phase1_gbm.py`

---

## Phase 2: Deep Learning Pipeline

> **Priority: MEDIUM** — Captures non-linear, long-range temporal dependencies.
> **Draws from:** Papers 1 (CNN+Transformer), 3 (CEEMDAN-GNN-Transformer), 10 (TCN-BiLSTM).

### 2.1 LSTM with Lag Window Input

The simplest deep baseline. Rather than single-step input, feed a **sliding window** of the last 48 hours as a sequence.

```
Input:  [PM10(t-47), ..., PM10(t-1), met_vars(t-47...t-1)] → shape (batch, 48, num_features)
Output: PM10(t+1)  — 1h ahead
```

- Train with Adam optimizer, early stopping, and dropout regularization.
- Compare against SARIMAX and GBM baselines.
- **Implementation file:** `analysis/phase2_lstm.py`

### 2.2 TCN-BiLSTM with STL Decomposition (Paper 10 Replication)

This is the architecture validated in Paper 10 (TCN-BiLSTM-DMAttention) that achieved a 49% average MAPE reduction. Adapt it to the Slovenian dataset:

```
Raw PM10 → STL Decompose →
  Trend Component     → TCN → BiLSTM → Attention → Trend Pred
  Seasonal Component  → TCN → BiLSTM → Attention → Seasonal Pred
  Residual Component  → TCN → BiLSTM → Attention → Residual Pred
                                                  → Sum → Final PM10 Pred
```

- Use `pytorch` or `tensorflow/keras`.
- The Dependency Matrix Attention (DMAttention) from Paper 10 can be simplified to a standard Bahdanau-style attention for a first pass.
- **Implementation file:** `analysis/phase2_tcn_bilstm.py`

### 2.3 Transformer-Based Model (Optional Advanced)

If the TCN-BiLSTM shows gains, add a Transformer encoder (like Paper 1's CNN+Transformer) for comparison:

```
Input Window → Patch Embedding → Transformer Encoder (Self-Attention) → FC Head → PM10 Pred
```

This handles long-range dependencies better than LSTM for windows > 48 hours.

---

## Phase 3: Spatial Modelling (Graph Neural Network)

> **Priority: MEDIUM-LOW** — Significant accuracy gain for multi-station products.
> **Draws from:** Papers 4 (ST-GNN for PM2.5), 5 (ST-Field NN), 6 (Graph Feature Fusion).

### 3.1 Build the Station Graph

Construct a graph where each of the 21 stations is a node, and edges are weighted by **geographic distance** (inverse-distance weighting):

```python
import networkx as nx
from sklearn.metrics.pairwise import haversine_distances

# Compute pairwise haversine distances between station lat/lon
# Create edge weights = 1 / distance (closer stations = stronger edge)
# Build adjacency matrix A (21 × 21)
```

> **Note:** Station lat/lon coordinates must be sourced from ARSO (see FINDINGS.md). This step requires resolving all 21 station codes to geographic coordinates (currently only E421 is confirmed as Nova Gorica / Grčna).

### 3.2 Spatio-Temporal GNN (Paper 6 Approach)

Following Paper 6's directed graph approach (18.65% MAE improvement for 24h forecasting):

```
For each timestep t:
  Node features: [PM10(t), PM2.5(t), temp(t), wind(t), pressure(t)] per station
  Graph Conv → Captures spatial dependencies from neighbouring stations
  Temporal Module (GRU or LSTM) → Captures temporal evolution at each node
  Output: PM10(t+1) for all 21 stations simultaneously
```

- Use `torch_geometric` or `dgl` (Deep Graph Library).
- Loss = mean MSE across all station outputs.
- **Implementation file:** `analysis/phase3_stgnn.py`

### 3.3 Air Quality Inference for Unmonitored Locations (Paper 5 Approach)

Once the ST-GNN is trained, use the Paper 5 (ST-Field NN) idea to estimate PM10 at arbitrary lat/lon coordinates in Slovenia beyond the 21 monitored stations. This would be the strongest differentiator for a product (e.g., "what's the air quality in my neighbourhood, which has no nearby station?").

---

## Evaluation Framework

All models should be evaluated on a **consistent, comparable basis**:

| Metric | Formula | Why It Matters |
|--------|---------|----------------|
| **MAE** | mean(\|y - ŷ\|) | Interpretable in µg/m³ units |
| **RMSE** | √mean((y - ŷ)²) | Penalizes large errors (pollution spikes) |
| **MAPE** | mean(\|y - ŷ\| / y) × 100 | Scale-free; comparable across stations |
| **POD** | TP / (TP + FN) | Fraction of pollution events correctly caught |
| **FAR** | FP / (FP + TN) | False alarm rate — critical for alerting products |

**Evaluation Protocol:**
- Train: first 80% of time series (chronological)
- Test: last 20%
- Forecast horizons to evaluate: **1h, 6h, 12h, 24h, 48h** ahead
- Evaluate on **all 21 stations**; report mean ± std across stations

---

## Proposed Experiment Comparison Table

| Model | Type | Key Innovation | Expected Improvement |
|-------|------|----------------|----------------------|
| Naive Persistence | Baseline | Next hour = current | — |
| SARIMAX | Classical | Seasonal ARIMA + covariates | Moderate over naive |
| RF + Lag Features | ML (fixed) | Lags + rolling windows | Should beat naive now |
| XGBoost / LightGBM | ML | Gradient boosting on lag features | Best among traditional ML |
| LSTM (sliding window) | Deep Learning | Sequence modelling | Better than GBM at 24h+ horizons |
| STL + TCN-BiLSTM | Deep Learning | Decomposition + hybrid conv-recurrent | 40–50% MAPE gain expected (per Paper 10) |
| ST-GNN | Spatial DL | Graph-based multi-station modelling | Best for multi-station + 24h forecasting |

---

## File & Directory Structure

```
analysis/
├── AQI_Analysis.ipynb          # Existing — fix Phase 0 here
├── phase1_stl.py               # STL decomposition + SARIMAX
├── phase1_gbm.py               # XGBoost / LightGBM with lag features
├── phase2_lstm.py              # Sliding-window LSTM
├── phase2_tcn_bilstm.py        # TCN-BiLSTM with STL (Paper 10 approach)
├── phase3_stgnn.py             # Spatio-Temporal GNN (Paper 6 approach)
└── utils/
    ├── lag_features.py         # Lag + rolling feature engineering
    ├── stl_decompose.py        # STL decomposition helpers
    ├── metrics.py              # MAE, RMSE, MAPE, POD, FAR
    └── graph_builder.py        # Station graph construction from lat/lon
images/
├── (existing plots)
├── stl_decomposition_E421.png
├── model_comparison_1h.png
├── model_comparison_24h.png
└── spatial_prediction_map.png
```

---

## Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Station lat/lon not fully mapped | Research remaining ARSO station codes; use E421 (confirmed) as starting point |
| Deep models overfit on ~14,000 timesteps per station | Use dropout, early stopping, and walk-forward cross-validation |
| CEEMDAN / STL decomposition increases pipeline complexity | Start with STL only (simpler); add CEEMDAN only if STL proves insufficient |
| GNN requires all 21 station coordinates | Fallback: use Euclidean distance approximation from station ID patterns if lat/lon is unavailable |
| `precipitation` column semantics unresolved | Exclude from modelling until semantics are empirically confirmed; test both with/without |

---

## References (from Literature Review)
- **Paper 1** — Next-Gen AQI Forecasting with CNN+Transformer (IJETT, 2025)
- **Paper 3** — CEEMDAN-GNN-Transformer (Frontiers in Env. Sci., 2025)
- **Paper 4** — ST-GNN for PM2.5 (MDPI Atmosphere, 2026)
- **Paper 5** — ST-Field NN for Air Quality Inference (IJCAI, 2024)
- **Paper 6** — Spatio-Temporal Feature Fusion over Graphs (MDPI Processes, 2025)
- **Paper 10** — TCN-BiLSTM-DMAttention with STL (Scientific Reports, 2023)
