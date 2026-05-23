"""
Email alert module — Gmail SMTP.
Credentials must live in .streamlit/secrets.toml only, never hardcoded.
One alert per calendar day enforced via data/last_alert_date.txt.
"""
import os
import smtplib
from datetime import date, datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from src.config import DATA_DIR

_LAST_ALERT_FILE = os.path.join(DATA_DIR, "last_alert_date.txt")


def already_sent_today() -> bool:
    if not os.path.exists(_LAST_ALERT_FILE):
        return False
    try:
        with open(_LAST_ALERT_FILE) as f:
            return f.read().strip() == date.today().isoformat()
    except Exception:
        return False


def _record_alert_sent() -> None:
    os.makedirs(os.path.dirname(_LAST_ALERT_FILE), exist_ok=True)
    with open(_LAST_ALERT_FILE, "w") as f:
        f.write(date.today().isoformat())


def send_signal_alert(
    signal: dict,
    regime_info: dict | None,
    sender_email: str,
    app_password: str,
    recipient_email: str,
) -> tuple[bool, str]:
    """Send one Gmail SMTP alert. Returns (success, message)."""
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

    subject = f"Gold AI Alert — {sig_lbl} Signal Detected"
    body = f"""\
Gold AI Decision Intelligence — Signal Alert
{'='*52}

Signal Direction:   {sig_lbl}
Model Confidence:   {conf_pct:.1f}%
Gold Price (XAU):   ${cur_price:,.2f}
Timestamp (UTC):    {ts}

Directional Probabilities
  DOWN:             {prob_down}
  SIDEWAYS:         {prob_sideways}
  UP:               {prob_up}

Market Regime:      {regime_lbl}
ATR:                {atr_str}

{'='*52}
⚠️  This is an automated research alert only.
    Not financial advice. Past performance does
    not guarantee future results.

Gold AI Decision Intelligence Platform
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
        _record_alert_sent()
        return True, "sent"
    except Exception as exc:
        return False, str(exc)
