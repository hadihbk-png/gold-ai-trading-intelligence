# APEX Metals AI

A multi-page Streamlit app for precious metals (XAU, XAG, XPT) directional trading intelligence — ML ensemble signals, walk-forward validation, risk management, and live backtesting. Built for personal research, not financial advice.

**[Live App →](https://apex-metals-ai.streamlit.app/)**

---

## Features

### Phase 1 — Core ML Pipeline
- Daily OHLCV ingestion for `GC=F` plus correlated assets (GLD, DXY, TNX, VIX, SI=F, CL=F, SPY) via yfinance
- Technical indicators: RSI, MACD, Bollinger Bands, ATR, SMA/EMA, lag features
- XGBoost + LightGBM + CatBoost stacking ensemble with Optuna hyperparameter tuning
- OOF (out-of-fold) Level-2 stacking — no data leakage
- Calibrated directional probabilities (UP / SIDEWAYS / DOWN)
- ATR-based walk-forward backtester: 2:1 R/R, slippage, spread, overnight financing
- Benchmark comparison: RSI, MA crossover, buy-and-hold, VIX-gated

### Phase 2 — Intelligence & Alerts
- Live gold spot price feed (Twelve Data with yfinance fallback)
- AI 5-day price projection chart
- Signal strength indicator panel
- Gmail SMTP email alerts: SIGNAL (once/day) and RISK (4h cooldown, max 3/day)
- Risk alert status indicator on Dashboard
- Extended Statistical Validation panel (Brier score, reliability curves, calibration)
- FRED macro data integration: yield curve, real rates, CPI YoY, Fed Funds gap

### Phase 3 — Prediction Intelligence Upgrade
- Feature enrichment: economic calendar proximity features (FOMC, CPI, NFP)
- Multi-timeframe architecture: weekly trend features + 4H confirmation signal
- Market regime detection (6 regimes) with macro regime gate
- Walk-forward validation panel (252-bar train / 21-bar test windows)
- Auto-retrain on stale model artifacts (Streamlit Cloud compatible, target <50s)
- Full backtesting panel with regime breakdown, directional accuracy, and benchmark comparison

### Phase 4 — Data Integrity & Claude AI
- Multi-source price waterfall: metals.live / Alpha Vantage / Twelve Data / yfinance
- LBMA GLD proxy and COMEX GC=F references
- Multi-currency display (AED default, 9 currencies)
- Transparency panel: data sources, variance analysis, market hours, FX rates
- Claude AI Morning Brief and signal explanation (claude-sonnet-4-6)

### Phase 6 — APEX Metals AI Expansion (in progress)
- Landing page with platform overview
- Silver (XAG/USD) and Platinum (XPT/USD) signal models
- Multi-metal comparison panel and portfolio tracker
- Decision Intelligence Centre: GO/CAUTION/NO-TRADE verdict
- ATR-based research zones
- SHAP transparency layer
- KPI Command Centre with audit trail

---

## Pages

| Page | Description |
|---|---|
| Dashboard | Live signal, regime, spot price, 5-day projection, retrain controls |
| Risk Management | Position sizing, drawdown controls, risk alert configuration |
| Historical Performance | Backtest results, benchmark comparison, equity curve |
| Live Validation | Walk-forward validation, calibration, statistical validation |

---

## Tech Stack

| Layer | Libraries |
|---|---|
| App framework | Streamlit ≥ 1.35 |
| ML models | XGBoost, LightGBM, CatBoost |
| Tuning | Optuna |
| Data | yfinance, fredapi, pandas, numpy |
| Calibration / metrics | scikit-learn, imbalanced-learn, scipy |
| Charts | Plotly |
| AI | Anthropic Claude API (claude-sonnet-4-6) |
| Alerts | Gmail SMTP (smtplib) |

---

## Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional — add to `.streamlit/secrets.toml`:

```toml
FRED_API_KEY   = "your_fred_key"       # macro features (FRED)
NEWSAPI_KEY    = "your_newsapi_key"    # sentiment (optional)
TWELVE_DATA_API_KEY = "your_td_key"   # live spot price
ANTHROPIC_API_KEY = "your_key"        # Claude AI morning brief
email_sender   = "you@gmail.com"
email_password = "app_password"
email_recipient = "you@gmail.com"
```

## Streamlit Cloud Deployment

Set `app.py` as the entry point. The app loads pre-trained artifacts from `models/models.pkl` and `data/raw_data.pkl` at startup. Auto-retrain triggers automatically when artifacts are stale (>3 days). Add the secrets above in the Streamlit Cloud dashboard under **Settings → Secrets**.

---

## Data Integrity

- All indicators computed from current and prior bars only (no look-ahead)
- Target is `Close.shift(-1)` — next-day price is never used as a feature
- Train/test split is strictly time-ordered
- OOF stacking uses `TimeSeriesSplit` — no shuffled cross-validation
- External ticker and macro features are lagged before modelling
- Walk-forward windows are non-overlapping and sequential

---

> **Disclaimer:** For personal research only. Not financial advice. Past performance does not guarantee future results.
