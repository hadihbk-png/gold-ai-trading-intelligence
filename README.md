# Gold AI Trader Assistant

A Streamlit MVP for gold price intelligence, directional model signals, risk sizing, and walk-forward validation reporting.

## Disclaimer

This tool is for personal research only. It does not constitute financial advice. Past performance does not guarantee future results.

## Deployment Status

Gold AI v1 is frozen for demo/deployment. The app loads saved artifacts from `models/models.pkl`; Streamlit Cloud should not need to retrain the model at startup.

Frozen walk-forward validation summary:

- Profitable windows: 5 / 6
- Average AI return: +0.58%
- Average Sharpe: 0.409
- Average max drawdown: -0.85%
- Average win rate: 47.2%

## Features

- Data: daily OHLCV for `GC=F` plus GLD, DXY, TNX, VIX, silver, crude oil, and SPY features via yfinance.
- Indicators: RSI, MACD, Bollinger Bands, ATR, SMA/EMA trend features, returns, lag features, event features, and weekly trend features.
- Models: XGBoost, LightGBM, and CatBoost with stacking and calibrated directional probabilities.
- Backtesting: ATR-based stops, 2:1 reward/risk, transaction costs, slippage, drawdown controls, and benchmark comparison.
- Pages: Dashboard, Risk Management, and Historical Performance.

## Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud

Use `app.py` as the main file.

The repository should include:

- `requirements.txt`
- `app.py`
- `pages/`
- `src/`
- `data/raw_data.pkl`
- `models/models.pkl`
- `run_v1_sanity_check.log`

Optional Streamlit secrets:

- `FRED_API_KEY`
- `NEWSAPI_KEY`

## Usage

1. Run `streamlit run app.py`.
2. The Dashboard auto-loads cached market data and saved model artifacts.
3. Review the current signal, risk levels, and historical validation pages.
4. Use **Train** only for local experimentation; the deployed demo should use the committed frozen model artifact.

## Anti-Leakage Checklist

- Indicators use rolling/EWM calculations derived from current and prior bars only.
- Target is `Close.shift(-1)`, so next-day price is never used as a feature.
- Train/test split is strictly time ordered.
- Optuna validation uses the later part of the training window.
- External ticker and macro features are lagged before modelling.
