# Stock-Market Prediction · GRU

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Task](https://img.shields.io/badge/task-time_series_forecasting-4a4a52?style=flat-square)

Forecasting stock-price movement from historical series with a **GRU** (Gated Recurrent Unit)
network. GRUs keep the long-memory behaviour of an LSTM with fewer gates, which makes them a
tidy fit for windowed price sequences.

## What's inside

| File | Role |
|------|------|
| `StockMarketPredictionGRU.ipynb` | Full notebook: load prices → window → scale → GRU → forecast & plot |
| `checktfgpu.ipynb` | Quick sanity check that TensorFlow sees the GPU |

## Pipeline

1. **Window**: slice the price history into fixed-length look-back sequences → next-step target.
2. **Scale**: `MinMax` normalise so the network trains on a stable range; keep the scaler to invert predictions.
3. **Model**: stacked `GRU` layers with dropout and a dense output, trained on mean-squared error.
4. **Evaluate**: plot predicted vs. actual on a held-out tail of the series.

## Run it

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
jupyter notebook StockMarketPredictionGRU.ipynb
```

> Learning project, not financial advice. See the notebook for the window size, architecture, and measured error.
