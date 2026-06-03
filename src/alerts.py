"""
Email alert module — Gmail SMTP.
Credentials must live in .streamlit/secrets.toml only, never hardcoded.

Alert types
-----------
SIGNAL  One per calendar day.  High-confidence directional signal.
        Tracking file: data/last_alert_date.txt

RISK    4-hour cooldown, max 3 per calendar day.
        Fires when >= 2 of 4 risk conditions are met simultaneously.
        Tracking file: data/risk_alert_state.json
"""
import json
import os
import smtplib
from datetime import date, datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from src.config import DATA_DIR

# ── File paths ─────────────────────────────────────────────────────────────────
_LAST_ALERT_FILE = os.path.join(DATA_DIR, "last_alert_date.txt")
_RISK_STATE_FILE = os.path.join(DATA_DIR, "risk_alert_state.json")

_UAE_TZ              = timezone(timedelta(hours=4))   # Gulf Standard Time (UTC+4)
_RISK_COOLDOWN_HOURS = 4
_RISK_MAX_PER_DAY    = 3


# ── Internal helper ────────────────────────────────────────────────────────────
def _safe_float(val, default: float = 0.0) -> float:
    """Convert val to float, returning default on NaN / None / error."""
    try:
        f = float(val)
        return default if f != f else f   # f != f is True only for NaN
    except (TypeError, ValueError):
        return default


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL ALERT  (one per calendar day)
# ══════════════════════════════════════════════════════════════════════════════

def already_sent_today() -> bool:
    if not os.path.exists(_LAST_ALERT_FILE):
        return False
    try:
        with open(_LAST_ALERT_FILE) as f:
            return f.read().strip() == date.today().isoformat()
    except Exception:
        return False


def _record_signal_sent() -> None:
    os.makedirs(os.path.dirname(_LAST_ALERT_FILE), exist_ok=True)
    with open(_LAST_ALERT_FILE, "w") as f:
        f.write(date.today().isoformat())


def send_signal_alert(
    signal: dict,
    regime_info: dict | None,
    sender_email: str,
    app_password: str,
    recipient_email: str,
    aed_price: float | None = None,
    price_source: str = "",
) -> tuple[bool, str]:
    """Send one Gmail SMTP signal alert. Returns (success, message)."""
    if already_sent_today():
        return False, "already_sent_today"

    sig_lbl    = signal.get("signal_label", "UNKNOWN")
    conf_pct   = signal.get("confidence_pct", 0)
    cur_price  = signal.get("current_price", 0)
    proba      = signal.get("proba_vec") or []
    atr_val    = signal.get("atr")
    regime_lbl = regime_info["regime_label"] if regime_info else "Unknown"
    ts         = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    prob_down     = f"{proba[0]*100:.1f}%" if len(proba) > 0 else "—"
    prob_sideways = f"{proba[1]*100:.1f}%" if len(proba) > 1 else "—"
    prob_up       = f"{proba[2]*100:.1f}%" if len(proba) > 2 else "—"
    atr_str       = f"${atr_val:.2f}" if atr_val is not None else "—"
    aed_str       = f" / AED {aed_price:,.0f}" if aed_price else ""
    src_str       = price_source or "Market Data"

    subject = f"APEX Metals AI Alert — {sig_lbl} — ${cur_price:,.0f}{aed_str}"
    body = f"""\
APEX Metals AI — Signal Alert
{'='*52}

Signal Direction:   {sig_lbl}
Model Confidence:   {conf_pct:.1f}%
Gold Price (XAU):   ${cur_price:,.2f}{aed_str}
Price Source:       {src_str}
Timestamp (UTC):    {ts}

Directional Probabilities
  DOWN:             {prob_down}
  SIDEWAYS:         {prob_sideways}
  UP:               {prob_up}

Market Regime:      {regime_lbl}
ATR:                {atr_str}

{'='*52}
Price data: Kitco Spot · Benchmark: LBMA
All data sourced from independent global
market data providers.

This is an automated research alert only.
Not financial advice. Past performance does
not guarantee future results.

APEX Metals AI
"""
    msg = MIMEMultipart()
    msg["From"]    = sender_email
    msg["To"]      = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(sender_email, app_password)
            smtp.sendmail(sender_email, recipient_email, msg.as_string())
        _record_signal_sent()
        return True, "sent"
    except Exception as exc:
        return False, str(exc)


# ══════════════════════════════════════════════════════════════════════════════
# RISK ALERT  (4-hour cooldown, max 3/day)
# ══════════════════════════════════════════════════════════════════════════════

def _read_risk_state() -> dict:
    """Load risk state JSON; auto-reset today_count on a new calendar day."""
    blank = {"today_date": "", "today_count": 0, "last_sent_utc": ""}
    if not os.path.exists(_RISK_STATE_FILE):
        return blank
    try:
        with open(_RISK_STATE_FILE) as f:
            data = json.load(f)
        if data.get("today_date") != date.today().isoformat():
            return {**blank, "today_date": date.today().isoformat()}
        return {**blank, **data}
    except Exception:
        return blank


def _write_risk_state(state: dict) -> None:
    os.makedirs(os.path.dirname(_RISK_STATE_FILE), exist_ok=True)
    with open(_RISK_STATE_FILE, "w") as f:
        json.dump(state, f)


def risk_alert_eligible() -> tuple[bool, str]:
    """
    Return (True, 'eligible') if both cooldown and daily cap allow a send.
    Return (False, reason_string) otherwise.
    """
    state = _read_risk_state()
    today = date.today().isoformat()

    # Daily cap check
    if state.get("today_date") == today and state.get("today_count", 0) >= _RISK_MAX_PER_DAY:
        return False, f"daily_cap_reached ({_RISK_MAX_PER_DAY}/day)"

    # 4-hour cooldown check
    last_sent = state.get("last_sent_utc", "")
    if last_sent:
        try:
            last_dt  = datetime.fromisoformat(last_sent).replace(tzinfo=timezone.utc)
            elapsed  = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
            if elapsed < _RISK_COOLDOWN_HOURS:
                remaining = _RISK_COOLDOWN_HOURS - elapsed
                return False, f"cooldown ({remaining:.1f}h remaining)"
        except Exception:
            pass

    return True, "eligible"


def evaluate_risk_conditions(
    df,
    cur_price: float,
    signal: dict | None = None,
) -> tuple[list[tuple[str, str]], float, float, float]:
    """
    Evaluate the four risk conditions against the latest bar in df.

    Parameters
    ----------
    df         : pandas DataFrame with feature columns (RSI, BB_*, Open, …)
    cur_price  : live or last-known gold price
    signal     : current signal dict (optional — used for conflict detection)

    Returns
    -------
    triggered     : list of (condition_key, human_readable_label)
    rsi           : latest RSI value
    bb_pct_b      : latest BB %B  (0 = lower band, 1 = upper band)
    intraday_pct  : % move from daily open to cur_price
    """
    last = df.iloc[-1]

    rsi       = _safe_float(last.get("RSI"),       default=50.0)
    bb_upper  = _safe_float(last.get("BB_Upper"),  default=None) if "BB_Upper" in df.columns else None
    bb_lower  = _safe_float(last.get("BB_Lower"),  default=None) if "BB_Lower" in df.columns else None
    bb_pct_b  = _safe_float(last.get("BB_PctB"),   default=0.5)
    open_price = _safe_float(last.get("Open"),     default=cur_price)
    intraday_pct = (cur_price - open_price) / open_price * 100 if open_price else 0.0

    triggered: list[tuple[str, str]] = []

    # ── Condition 1: RSI extreme ───────────────────────────────────────────────
    if rsi > 70:
        triggered.append(("RSI_OVERBOUGHT", f"RSI {rsi:.1f} — Overbought (>70)"))
    elif rsi < 30:
        triggered.append(("RSI_OVERSOLD", f"RSI {rsi:.1f} — Oversold (<30)"))

    # ── Condition 2: Bollinger Band breach ────────────────────────────────────
    if bb_upper is not None and cur_price > bb_upper:
        triggered.append(("BB_UPPER", f"Price ${cur_price:,.2f} above Upper BB ${bb_upper:,.2f}"))
    elif bb_lower is not None and cur_price < bb_lower:
        triggered.append(("BB_LOWER", f"Price ${cur_price:,.2f} below Lower BB ${bb_lower:,.2f}"))

    # ── Condition 3: Intraday move > 1.5% from daily open ────────────────────
    if abs(intraday_pct) > 1.5:
        side = "above" if intraday_pct > 0 else "below"
        triggered.append(("INTRADAY", f"Price {intraday_pct:+.2f}% {side} daily open ${open_price:,.2f}"))

    # ── Condition 4: Signal direction conflicts with intraday price move ──────
    if signal:
        sig_int = signal.get("signal_int", 1)
        if sig_int == 2 and intraday_pct < -0.3:
            triggered.append(("SIGNAL_CONFLICT",
                               f"UP signal conflicts with intraday decline {intraday_pct:+.2f}%"))
        elif sig_int == 0 and intraday_pct > 0.3:
            triggered.append(("SIGNAL_CONFLICT",
                               f"DOWN signal conflicts with intraday rise {intraday_pct:+.2f}%"))

    return triggered, rsi, bb_pct_b, intraday_pct


def _suggested_action(triggered_keys: list[str]) -> str:
    """Single suggested action derived from the set of triggered condition keys."""
    if "RSI_OVERBOUGHT" in triggered_keys or "BB_UPPER" in triggered_keys:
        return "Reduce Exposure"
    if "INTRADAY" in triggered_keys:
        return "Tighten Stop-Loss"
    return "Monitor Closely"


def send_risk_alert(
    df,
    cur_price: float,
    signal: dict | None,
    sender_email: str,
    app_password: str,
    recipient_email: str,
) -> tuple[bool, str]:
    """
    Evaluate risk conditions and, if >= 2 are triggered and cooldown allows,
    send a RISK Management alert email.
    Returns (sent: bool, reason: str).
    """
    eligible, reason = risk_alert_eligible()
    if not eligible:
        return False, reason

    triggered, rsi, bb_pct_b, intraday_pct = evaluate_risk_conditions(df, cur_price, signal)

    if len(triggered) < 2:
        return False, f"insufficient_conditions ({len(triggered)}/2 required)"

    # ── Build email content ────────────────────────────────────────────────────
    now_utc  = datetime.now(timezone.utc)
    now_uae  = datetime.now(_UAE_TZ)
    ts_uae   = now_uae.strftime("%Y-%m-%d %H:%M GST")
    ts_utc   = now_utc.strftime("%Y-%m-%d %H:%M UTC")

    last       = df.iloc[-1]
    open_price = _safe_float(last.get("Open"), default=cur_price)
    daily_chg  = (cur_price - open_price) / open_price * 100 if open_price else 0.0

    # RSI status label
    if rsi > 70:
        rsi_status = "OVERBOUGHT"
    elif rsi < 30:
        rsi_status = "OVERSOLD"
    else:
        rsi_status = "Neutral"

    # BB %B position label
    if bb_pct_b > 1.0:
        bb_pos = f"%B {bb_pct_b:.2f} — Above Upper Band"
    elif bb_pct_b < 0.0:
        bb_pos = f"%B {bb_pct_b:.2f} — Below Lower Band"
    elif bb_pct_b >= 0.8:
        bb_pos = f"%B {bb_pct_b:.2f} — Near Upper Band"
    elif bb_pct_b <= 0.2:
        bb_pos = f"%B {bb_pct_b:.2f} — Near Lower Band"
    else:
        bb_pos = f"%B {bb_pct_b:.2f} — Mid-range"

    triggered_keys  = [k for k, _ in triggered]
    action          = _suggested_action(triggered_keys)
    bullet_conds    = "\n".join(f"  * {lbl}" for _, lbl in triggered)
    cond_summary    = " + ".join(lbl.split("—")[0].strip() for _, lbl in triggered[:2])

    subject = f"GOLD RISK ALERT — {cond_summary} — ${cur_price:,.2f} — {ts_uae}"

    body = f"""\
{'='*52}
  RISK MANAGEMENT ALERT
{'='*52}

1. PRICE SUMMARY
   Current Price:    ${cur_price:,.2f}
   Daily Open:       ${open_price:,.2f}
   Change from Open: {daily_chg:+.2f}%
   Time (UAE/GST):   {ts_uae}
   Time (UTC):       {ts_utc}

2. RSI
   RSI Value:        {rsi:.1f}
   Status:           {rsi_status}

3. BOLLINGER BANDS
   Position:         {bb_pos}

4. TRIGGERED CONDITIONS
{bullet_conds}

5. SUGGESTED ACTION
   >>> {action} <<<

{'='*52}
This is an automated research alert only.
Not financial advice. Past performance does
not guarantee future results.

APEX Metals AI
"""

    msg = MIMEMultipart()
    msg["From"]    = sender_email
    msg["To"]      = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(sender_email, app_password)
            smtp.sendmail(sender_email, recipient_email, msg.as_string())
    except Exception as exc:
        return False, str(exc)

    # Record successful send — increment daily count and timestamp
    state = _read_risk_state()
    today = date.today().isoformat()
    count = state.get("today_count", 0) + 1 if state.get("today_date") == today else 1
    _write_risk_state({
        "today_date":    today,
        "today_count":   count,
        "last_sent_utc": now_utc.isoformat(),
    })
    return True, "sent"
