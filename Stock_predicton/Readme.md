# MSFT Stock Price Prediction using LSTM

A deep learning project that uses Long Short-Term Memory (LSTM) neural networks to predict Microsoft (MSFT) stock closing prices — including a **30-day future forecast** with confidence bands.

---

## Dataset

| Property | Details |
|---|---|
| **File** | `MSFT.csv` |
| **Coverage** | 1986 – 2024 (9,636 daily rows) |
| **Training window** | 2010 – 2024 (~3,500 rows) |
| **Key columns** | `Date`, `Close`, `Volume` |

---

## Model Architecture

The model is a **stacked LSTM** built in PyTorch with the following design:

```
Input (60 timesteps × N features)
   └─► LSTM (2 layers, hidden_size=128, dropout=0.2)
         └─► Dropout(0.2)
               └─► Linear(128 → 64) + ReLU
                     └─► Linear(64 → 1)  →  Predicted Close Price
```

**Key design choices:**
- **Look-back window:** 60 trading days (~3 months) instead of the naive 3-day window
- **Stacked layers:** 2 LSTM layers to capture both short- and long-range dependencies
- **Dropout:** 20% between layers and before the FC head to regularise
- **Multi-feature input:** Close price is not the only signal — see Features below

---

## Features Used

| Feature | Description |
|---|---|
| `Close` | Daily closing price (primary target) |
| `EMA_20` | 20-day Exponential Moving Average |
| `RSI_14` | 14-day Relative Strength Index (via `ta` or manual) |
| `Volume` | Daily traded volume *(if present in CSV)* |

All features are normalised to **[0, 1]** with `MinMaxScaler` before training. A separate scaler is fitted on `Close` alone for clean inverse-transformation of predictions.

---

## Data Split

| Set | Share | Purpose |
|---|---|---|
| Train | 80% | Model fitting |
| Validation | 10% | Early stopping & LR scheduling |
| Test | 10% | Final held-out evaluation |

Temporal order is **always preserved** — no shuffle is applied.

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam (lr = 0.001) |
| Loss | MSELoss |
| Batch size | 32 |
| Max epochs | 300 |
| Early stopping patience | 20 epochs |
| LR scheduler | `ReduceLROnPlateau` (factor=0.5, patience=10) |
| Gradient clipping | max_norm = 1.0 |
| Random seed | 42 (reproducible) |

---

## Evaluation Metrics

Predictions are inverse-scaled back to USD before evaluation:

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error (dollars off on average) |
| **RMSE** | Root Mean Squared Error (penalises large errors) |
| **MAPE** | Mean Absolute Percentage Error (scale-independent %) |

Metrics are reported for **Train**, **Validation**, and **Test** sets separately.

---

## 30-Day Future Forecast

After training, the model performs **recursive autoregressive forecasting**:

1. The last 60-day window from the dataset is fed to the model.
2. Each predicted Close price is appended back into the sliding window.
3. This repeats for 30 business days forward.
4. A **±1 standard deviation confidence band** (based on recent price volatility) is drawn around the forecast line.

The forecast is displayed as an interactive **Plotly** chart with a range-slider and quick-zoom buttons (1M / 3M / 6M / All).

---

## Notebook Walkthrough

| Step | Description |
|---|---|
| 1 | Import libraries & set device (CPU / CUDA) |
| 2 | Load and inspect `MSFT.csv` |
| 3 | Filter to 2010 – 2024 for a richer training signal |
| 4 | Feature engineering — EMA-20, RSI-14, Volume |
| 5 | Normalise with MinMaxScaler |
| 6 | Build 60-step windowed sequences |
| 7 | Train / Validation / Test split |
| 8 | Define `ImprovedLSTM` model |
| 9 | Wrap data in PyTorch `DataLoader` |
| 10 | Train with early stopping & LR scheduler |
| 11 | Plot loss curves |
| 12 | Generate predictions & inverse-scale to USD |
| 13 | Print MAE / RMSE / MAPE for all splits |
| 14–17 | Matplotlib plots (Train, Val, Test, Combined) |
| 18 | 30-day future forecast with confidence band (Plotly) |
| 19 | Interactive Plotly chart — all splits + forecast |
| 20 | Recursive predictions on Val + Test sets |
| 21 | Final evaluation metrics on test set |

---

## Installation

```bash
pip install torch numpy pandas scikit-learn matplotlib plotly ta
```

> The `ta` library is optional. If not installed, RSI-14 is computed manually using a rolling-window approach.

---

## Usage

1. Place `MSFT.csv` in the same directory as the notebook.
2. Open `Stock_prediction_LSTM.ipynb` in Jupyter.
3. Run all cells top-to-bottom (`Kernel → Restart & Run All`).

For GPU acceleration, ensure a CUDA-capable device is available — the notebook detects it automatically via `torch.cuda.is_available()`.

---

## Disclaimer

This project is for **educational and research purposes only**. Stock price predictions made by this model should **not** be used as the basis for real financial decisions. Past performance does not guarantee future results.
