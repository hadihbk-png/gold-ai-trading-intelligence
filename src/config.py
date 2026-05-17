import os
from datetime import datetime, timedelta

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ── Price tickers ──────────────────────────────────────────────────────────────
PRIMARY_TICKER = "GC=F"
TICKERS = [
    "GC=F",       # Gold Futures (primary)
    "GLD",        # Gold ETF (replaces XAUUSD=X which was delisted from Yahoo)
    "DX-Y.NYB",   # US Dollar Index
    "^TNX",       # 10-Year Treasury Yield
    "^VIX",       # CBOE VIX
    "SI=F",       # Silver Futures
    "CL=F",       # Crude Oil Futures
    "SPY",        # S&P 500 ETF
]

# ── Date ranges ────────────────────────────────────────────────────────────────
END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=365 * 5 + 90)).strftime("%Y-%m-%d")
TRAIN_YEARS = 4
TEST_YEARS  = 1

# ── FRED macro data ────────────────────────────────────────────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
FRED_SERIES = {
    "FEDFUNDS":  "Fed Funds Rate",
    "DGS10":     "10Y Treasury Yield",
    "DGS2":      "2Y Treasury Yield",
    "T10YIE":    "10Y Breakeven Inflation",
    "CPIAUCSL":  "CPI All Urban Consumers",
    "UNRATE":    "Unemployment Rate",
}

# ── Technical indicator periods ────────────────────────────────────────────────
RSI_PERIOD  = 14
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9
BB_PERIOD   = 20
BB_STD      = 2
ATR_PERIOD  = 14
SMA_PERIODS = [20, 50, 200]
EMA_PERIODS = [20, 50]
LAG_PERIODS = [1, 2, 3, 5, 10, 20]

# ── Signal classification ──────────────────────────────────────────────────────
SIGNAL_THRESHOLD    = 0.003   # ±0.3% next-day return — kept for legacy compatibility
DIRECTION_THRESHOLD = 0.005   # ±0.5% next-day return → UP / DOWN / SIDEWAYS (primary target)

# ── Execution realism ──────────────────────────────────────────────────────────
TRANSACTION_COST   = 0.0008   # Broker commission per side (0.08%)
EXCHANGE_FEE       = 0.0001   # Exchange + clearing fee per side
BID_ASK_SPREAD     = 0.0002   # Estimated gold futures bid-ask half-spread
SLIPPAGE           = 0.0003   # Market impact / fill slippage
OVERNIGHT_RATE     = 0.00008  # Overnight financing per day (≈ 2% annualised / 252)

TOTAL_ENTRY_COST   = TRANSACTION_COST + EXCHANGE_FEE + BID_ASK_SPREAD + SLIPPAGE
TOTAL_EXIT_COST    = TRANSACTION_COST + EXCHANGE_FEE + BID_ASK_SPREAD + SLIPPAGE

# ── Position sizing & backtest ─────────────────────────────────────────────────
INITIAL_CAPITAL     = 100_000
RISK_PER_TRADE_PCT  = 0.01     # Risk 1% of capital per trade (ATR-based sizing)
ATR_STOP_MULTIPLIER = 2.0
RISK_REWARD_RATIO   = 2.0
MAX_OPEN_EXPOSURE   = 0.30     # Max 30% of capital at risk simultaneously
POSITION_SIZE       = 0.10     # Fallback fixed fraction (overridden by ATR sizing)

# ── Risk management (hard limits) ─────────────────────────────────────────────
MAX_DAILY_LOSS_PCT   = 0.02    # Halt trading if daily P&L < -2%
MAX_DRAWDOWN_HALT    = 0.15    # Halt trading if drawdown > 15%
DRAWDOWN_RESUME_PCT  = 0.10    # Resume when drawdown recovers to <10%

# ── No-Trade filters ──────────────────────────────────────────────────────────
MIN_CONFIDENCE       = 0.25    # Floor confidence — dynamic OOF thresholds act as primary gate
MAX_ATR_PCT          = 0.025   # Max ATR/price (2.5%) — suppress extreme volatility
MAX_SPREAD_PCT       = 0.0015  # Max estimated spread vs expected move

# ── Macro regime gate ─────────────────────────────────────────────────────────
# Applied in addition to the trend filter. Identifies hostile macro environments
# (rate shock, dollar surge, panic) and raises the confidence floor for trading.
# Does NOT suppress all trading — only tightens entry requirements.
REGIME_GATE_ENABLED   = True
GATE_TNX_SHOCK_PP     = 0.30   # TNX 20-bar rise > 0.30pp  → rate-shock regime
GATE_DXY_BULL_PCT     = 0.015  # DXY > 1.5% above SMA20   → strong-dollar regime
GATE_SHOCK_UP_CONF    = 0.55   # P(UP) floor in rate-shock + dollar-surge env
GATE_VIX_PANIC        = 30.0   # VIX level threshold for panic regime
GATE_ATR_REGIME_MAX   = 2.0    # ATR/ATR_MA63 threshold for extreme volatility
GATE_PANIC_CONF       = 0.60   # Confidence floor in panic regime (both directions)

# ── Range / choppy-market filter ───────────────────────────────────────────────
# Raises the confidence floor when ADX is below the trend threshold, suppressing
# low-conviction directional signals in non-trending / sideways environments.
ADX_RANGE_FILTER_ENABLED = True
ADX_RANGE_THRESHOLD      = 20.0   # ADX below this = ranging / non-trending
ADX_RANGE_CONF_FLOOR     = 0.55   # Min confidence required when market is ranging

# ── Trend regime filter ────────────────────────────────────────────────────────
# Suppresses counter-trend signals: DOWN signals in a confirmed bull regime
# require higher confidence, and UP signals in a confirmed bear regime likewise.
# Regime defined as: price > SMA_fast > SMA_slow AND SMA_fast slope is rising
# (measured over TREND_SLOPE_LOOKBACK bars).
TREND_FILTER_ENABLED  = True
TREND_BULL_DOWN_CONF  = 0.45   # P(DOWN) floor to allow a short in bull regime
TREND_BEAR_UP_CONF    = 0.45   # P(UP) floor to allow a long in bear regime
TREND_SMA_FAST        = 50     # Fast SMA period (matches SMA_PERIODS)
TREND_SMA_SLOW        = 200    # Slow SMA period (matches SMA_PERIODS)
TREND_SLOPE_LOOKBACK  = 20     # Bars over which SMA slope is measured

# ── Confirmed bull regime participation ────────────────────────────────────────
# When all six conditions below are met, the BUY confidence floor is relaxed
# from the OOF-tuned threshold (~0.44–0.47) to BULL_UP_CONF_RELAXED (~0.42),
# allowing more entries in genuine uptrends without touching bear/neutral logic.
# Conditions: price>SMA50>SMA200, SMA50 slope>0, VIX calm, TNX stable, DXY weak.
BULL_REGIME_ENABLED     = True
BULL_UP_CONF_RELAXED    = 0.42   # Lowered BUY floor in confirmed bull regime
BULL_VIX_CALM_THRESHOLD = 20.0   # VIX must be below this (calm environment)
BULL_TNX_STABLE_THRESH  = 0.15   # TNX 20-bar change < 0.15pp (not rate-shocked)
BULL_DXY_WEAK_THRESH    = 0.005  # DXY must be no more than 0.5% above its SMA20

# ── Confidence-based position sizing ──────────────────────────────────────────
CONF_SIZE_MIN = 0.5    # Scale at MIN_CONFIDENCE — half-size on marginal signals
CONF_SIZE_MAX = 1.0    # Scale at confidence=1.0 — full-size on high-conviction trades

# ── Dynamic exit logic ─────────────────────────────────────────────────────────
ATR_TRAIL_ENABLED       = False
ATR_TRAIL_MULTIPLIER    = 1.5    # Trail stop = price − 1.5×ATR (tighter than 2.0× entry)
CONF_DECAY_ENABLED      = False
CONF_DECAY_THRESHOLD    = 0.30   # Exit when P(trade direction) drops below this
EMA_TREND_EXIT_ENABLED  = False
EMA_TREND_EXIT_PERIOD   = 20     # EMA period for trend-reversal exit
EMA_TREND_EXIT_BUFFER   = 0.002  # Minimum entry margin above/below EMA to arm the exit

# ── Ensemble stacking ─────────────────────────────────────────────────────────
STACKING_CV_FOLDS    = 5       # TimeSeriesSplit folds for OOF generation
META_RIDGE_ALPHA     = 1.0     # Ridge regression regularisation
CALIBRATION_METHOD   = "isotonic"  # 'isotonic' or 'sigmoid'
CALIBRATION_CV_FOLDS = 3       # CV folds inside CalibratedClassifierCV

# ── Market regime detection ────────────────────────────────────────────────────
REGIME_LOOKBACK      = 63      # Rolling window (~3 months) for regime stats
VIX_HIGH_THRESHOLD   = 25.0
VIX_LOW_THRESHOLD    = 15.0
CPI_INFLATION_LEVEL  = 3.0     # CPI YoY% above which → inflationary regime
YIELD_CURVE_INVERT   = 0.0     # DGS10-DGS2 below this → risk-off

# ── Feature drift monitoring ───────────────────────────────────────────────────
DRIFT_PSI_THRESHOLD  = 0.20    # PSI > 0.2 → significant drift
DRIFT_MOD_THRESHOLD  = 0.10    # PSI 0.1-0.2 → moderate drift
DRIFT_WINDOW_DAYS    = 63      # Recent window to compare against training baseline

# ── Model hyperparameter search ────────────────────────────────────────────────
N_TRIALS     = 100
RANDOM_STATE = 42

# ── News sentiment (optional) ──────────────────────────────────────────────────
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

# ── 4H timeframe integration architecture ─────────────────────────────────────
# Design hook for future multi-timeframe execution layer:
# Daily model  → market direction + signal generation (current implementation)
# 4H model     → entry timing refinement (stub below, disabled by default)
# Combined     → daily_signal AND 4h_confirmation required to execute
TIMEFRAME_CONFIG = {
    "daily": {
        "enabled": True,
        "interval": "1d",
        "purpose": "direction_and_signal",
        "lookback_years": 5,
    },
    "intraday_4h": {
        "enabled": False,           # Flip to True when 4H module is implemented
        "interval": "1h",           # yfinance 1h data, then resample to 4H
        "purpose": "entry_timing",
        "lookback_days": 59,        # yfinance free-tier limit for 1h data
        "confirmation_required": True,
        "entry_delay_bars": 1,
    },
}
