"""
Four benchmark trading strategies.
All use the same run_backtest() engine for a fair apples-to-apples comparison.
"""

import numpy as np
import pandas as pd
from src.config import INITIAL_CAPITAL


def rsi_strategy(df_test: pd.DataFrame, oversold: float = 35.0, overbought: float = 65.0) -> np.ndarray:
    """Buy when RSI < oversold, Sell when RSI > overbought."""
    if "RSI" not in df_test.columns:
        return np.ones(len(df_test), dtype=int)
    rsi = df_test["RSI"]
    signals = np.where(rsi < oversold, 2, np.where(rsi > overbought, 0, 1))
    return np.nan_to_num(signals, nan=1).astype(int)


def ma_crossover_strategy(df_test: pd.DataFrame) -> np.ndarray:
    """Golden cross (SMA20 > SMA50) → Buy; death cross → Sell."""
    if "SMA_20" not in df_test.columns or "SMA_50" not in df_test.columns:
        return np.ones(len(df_test), dtype=int)
    signals = np.where(
        df_test["SMA_20"] > df_test["SMA_50"], 2,
        np.where(df_test["SMA_20"] < df_test["SMA_50"], 0, 1),
    )
    return np.nan_to_num(signals, nan=1).astype(int)


def breakout_strategy(df_test: pd.DataFrame, window: int = 20) -> np.ndarray:
    """Donchian channel breakout: close > 20-day high → Buy; close < 20-day low → Sell."""
    close = df_test["Close"]
    upper = close.rolling(window).max().shift(1)
    lower = close.rolling(window).min().shift(1)
    raw = np.where(close > upper, 2, np.where(close < lower, 0, 1))
    # First `window` bars have NaN channel levels → default to NO TRADE
    valid = (~upper.isna()).values
    signals = np.where(valid, raw, 1)
    return signals.astype(int)


def buy_hold_strategy(df_test: pd.DataFrame) -> np.ndarray:
    """Always hold long."""
    return np.full(len(df_test), 2, dtype=int)


def run_all_benchmarks(
    df: pd.DataFrame,
    test_dates: pd.DatetimeIndex,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    """
    Run all four benchmarks using the same backtest engine.

    Returns
    -------
    dict mapping strategy name → {equity, bh, trades, metrics} or {error}.
    """
    from src.backtest import run_backtest

    df_test = df.reindex(test_dates)

    strategies = {
        "RSI Strategy": rsi_strategy(df_test),
        "MA Crossover":  ma_crossover_strategy(df_test),
        "Breakout":      breakout_strategy(df_test),
        "Buy & Hold":    buy_hold_strategy(df_test),
    }

    results = {}
    for name, signals in strategies.items():
        try:
            eq, bh, trades, metrics = run_backtest(
                df, signals, test_dates, initial_capital=initial_capital,
            )
            results[name] = {
                "equity": eq, "bh": bh,
                "trades": trades, "metrics": metrics,
            }
        except Exception as exc:
            results[name] = {"error": str(exc)}

    return results


def benchmark_metrics_table(benchmark_results: dict) -> pd.DataFrame:
    """Format benchmark results as a comparison DataFrame."""
    rows = []
    for name, res in benchmark_results.items():
        if "error" in res:
            rows.append({"Strategy": name, "Error": res["error"]})
            continue
        m = res["metrics"]
        rows.append({
            "Strategy":        name,
            "Total Return (%)":  m.get("Total Return (%)", 0),
            "CAGR (%)":          m.get("CAGR (%)", 0),
            "Sharpe":            m.get("Sharpe Ratio", 0),
            "Sortino":           m.get("Sortino Ratio", 0),
            "Max DD (%)":        m.get("Max Drawdown (%)", 0),
            "Win Rate (%)":      m.get("Win Rate (%)", 0),
            "Profit Factor":     m.get("Profit Factor", 0),
            "Expectancy ($)":    m.get("Expectancy ($)", 0),
            "Trades":            m.get("Total Trades", 0),
        })
    return pd.DataFrame(rows).set_index("Strategy") if rows else pd.DataFrame()
