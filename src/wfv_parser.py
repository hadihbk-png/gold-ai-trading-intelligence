"""
Parse rolling walk-forward validation log files.
Returns structured window results and aggregate stats.
"""
import re
from pathlib import Path
from glob import glob
import os


def parse_validation_log(log_path: str) -> dict:
    """
    Parse a run_rolling_validation log file.
    Returns:
        {
          "windows": [{"label", "regime", "year", "return", "sharpe",
                       "sortino", "maxdd", "winrate", "pf", "trades",
                       "bh_return"}, ...],
          "aggregates": {"profitable_n", "total_n", "avg_return", ...},
          "elapsed_min": float,
          "ok": bool,
        }
    """
    try:
        content = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    except (FileNotFoundError, OSError):
        return {"ok": False, "windows": [], "aggregates": {}}

    if "VALIDATION COMPLETE" not in content:
        return {"ok": False, "windows": [], "aggregates": {}, "partial": True}

    windows = []
    curr: dict = {}
    state = "idle"

    for line in content.splitlines():
        stripped = line.strip()

        # ── Window header ──────────────────────────────────────────────────────
        wm = re.match(r"WINDOW\s+\d+/\d+\s*:\s*(.+?)\s*—\s*(.+)", stripped)
        if wm:
            curr = {"label": wm.group(1).strip(), "regime": wm.group(2).strip()}
            state = "window"
            continue

        if state == "idle":
            continue

        # Year from TEST line
        if "TEST   :" in line:
            ym = re.search(r"(\d{4})-\d{2}-\d{2}\s*→", line)
            if ym:
                curr["year"] = ym.group(1)

        # Enter backtest section
        if "Backtest results:" in stripped:
            state = "backtest"
            continue

        if state == "backtest":
            _metric_map = [
                ("return",  r"Total Return \(%\)\s*:\s*([-\d.]+)"),
                ("sharpe",  r"Sharpe Ratio\s*:\s*([-\d.]+)"),
                ("sortino", r"Sortino Ratio\s*:\s*([-\d.]+)"),
                ("maxdd",   r"Max Drawdown \(%\)\s*:\s*([-\d.]+)"),
                ("winrate", r"Win Rate \(%\)\s*:\s*([-\d.]+)"),
                ("pf",      r"Profit Factor\s*:\s*([\d.e+]+)"),
                ("trades",  r"Total Trades\s*:\s*(\d+)"),
                ("exp",     r"Expectancy \(\$\)\s*:\s*([-\d.]+)"),
            ]
            for key, pat in _metric_map:
                m = re.search(pat, stripped)
                if m:
                    curr[key] = float(m.group(1))

            if "Benchmark comparison:" in stripped:
                state = "benchmark"
            continue

        if state == "benchmark":
            if "Buy & Hold" in stripped:
                bm = re.search(r"Buy & Hold\s+([+-]\s*[\d.]+)%", stripped)
                if bm:
                    curr["bh_return"] = float(bm.group(1).replace(" ", ""))
                if "return" in curr:
                    windows.append(dict(curr))
                curr = {}
                state = "idle"
            continue

    # ── Aggregate stats ────────────────────────────────────────────────────────
    agg: dict = {}
    m = re.search(r"Profitable windows\s*:\s*(\d+)\s*/\s*(\d+)", content)
    if m:
        agg["profitable_n"] = int(m.group(1))
        agg["total_n"]      = int(m.group(2))

    for key, pat in [
        ("avg_return",  r"Avg AI return\s*:\s*([+-][\d.]+)%"),
        ("avg_bh",      r"Avg B&H return\s*:\s*([+-][\d.]+)%"),
        ("avg_sharpe",  r"Avg Sharpe\s*:\s*([-\d.]+)"),
        ("avg_maxdd",   r"Avg Max Drawdown\s*:\s*([-\d.]+)%"),
        ("avg_winrate", r"Avg Win Rate\s*:\s*([\d.]+)%"),
        ("avg_trades",  r"Avg Trades / window\s*:\s*([\d.]+)"),
    ]:
        m = re.search(pat, content)
        if m:
            agg[key] = float(m.group(1))

    elapsed = re.search(r"Total elapsed:\s*([\d.]+)\s*min", content)
    elapsed_min = float(elapsed.group(1)) if elapsed else 0.0

    return {"ok": True, "windows": windows, "aggregates": agg,
            "elapsed_min": elapsed_min}


def latest_validation_log(project_root: str) -> str | None:
    """Return the path of the most recently modified validation log."""
    patterns = [
        os.path.join(project_root, "run_v1_sanity_check.log"),
        os.path.join(project_root, "run_rolling_validation*.log"),
        os.path.join(project_root, "run_A_*.log"),
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob(pat))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)
