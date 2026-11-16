"""
TEMPORARY diagnostic — Platinum (XPT/USD) price-source probe.

Tests candidate platinum spot sources FROM THIS ENVIRONMENT (Streamlit Cloud)
and shows which return a usable price. Read-only — no app logic touched.
DELETE this page once we've wired the winning source.
"""
import streamlit as st
import requests

st.set_page_config(page_title="Price Probe", page_icon="🔬", layout="wide")
st.title("🔬 Platinum price-source probe (temporary)")
st.caption("Shows which XPT/USD sources work from this server. Delete after use.")


def _secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


_td = _secret("TWELVE_DATA_API_KEY")
_av = _secret("ALPHA_VANTAGE_API_KEY")

st.write(f"Twelve Data key present: **{bool(_td)}** · Alpha Vantage key present: **{bool(_av)}**")

results = []

# 1 — Twelve Data XPT/USD (your key)
try:
    r = requests.get("https://api.twelvedata.com/price",
                     params={"symbol": "XPT/USD", "apikey": _td}, timeout=8)
    results.append(("1. Twelve Data XPT/USD (your key)", f"HTTP {r.status_code}", r.text[:400]))
except Exception as e:
    results.append(("1. Twelve Data XPT/USD (your key)", "ERROR", str(e)[:400]))

# 2 — Swissquote public feed (keyless)
try:
    r = requests.get(
        "https://forex-data-feed.swissquote.com/public-quotes/bboquotes/instrument/XPT/USD",
        timeout=8)
    results.append(("2. Swissquote XPT/USD (keyless)", f"HTTP {r.status_code}", r.text[:500]))
except Exception as e:
    results.append(("2. Swissquote XPT/USD (keyless)", "ERROR", str(e)[:400]))

# 3 — metals.live platinum (keyless)
try:
    r = requests.get("https://api.metals.live/v1/spot/platinum", timeout=8)
    results.append(("3. metals.live platinum (keyless)", f"HTTP {r.status_code}", r.text[:300]))
except Exception as e:
    results.append(("3. metals.live platinum (keyless)", "ERROR", str(e)[:400]))

# 4 — Alpha Vantage XPT/USD (your key, if present)
if _av:
    try:
        r = requests.get("https://www.alphavantage.co/query",
                         params={"function": "CURRENCY_EXCHANGE_RATE",
                                 "from_currency": "XPT", "to_currency": "USD", "apikey": _av},
                         timeout=8)
        results.append(("4. Alpha Vantage XPT/USD (your key)", f"HTTP {r.status_code}", r.text[:400]))
    except Exception as e:
        results.append(("4. Alpha Vantage XPT/USD (your key)", "ERROR", str(e)[:400]))
else:
    results.append(("4. Alpha Vantage XPT/USD", "SKIPPED", "no ALPHA_VANTAGE_API_KEY in secrets"))

# 5 — yfinance PL=F (futures)
try:
    import pandas as pd
    import yfinance as yf
    d = yf.download("PL=F", period="5d", progress=False, auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    if not d.empty and "Close" in d.columns:
        close = d["Close"].dropna()
        price = float(close.iloc[-1]) if len(close) else None
        results.append(("5. yfinance PL=F", "OK" if price else "EMPTY", f"price = {price}"))
    else:
        results.append(("5. yfinance PL=F", "EMPTY", "no data returned"))
except Exception as e:
    results.append(("5. yfinance PL=F", "ERROR", str(e)[:400]))

st.subheader("Results")
for name, status, body in results:
    st.markdown(f"**{name}** — `{status}`")
    st.code(body or "(empty)", language="json")

st.info("Send these results to Claude. A valid price (e.g. ~1400-1600 for platinum) in any "
        "source tells us which to wire in. Then this page gets deleted.")
