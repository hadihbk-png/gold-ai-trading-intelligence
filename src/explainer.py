"""
Claude AI signal explanation and morning brief — Phase 4A Enhancement 3.

Two public functions:
  generate_signal_explanation(signal_data, api_key) → 150-200 word plain-English explanation
  generate_morning_brief(signal_data, date, api_key) → 200-250 word morning market brief

Both call the Anthropic Claude API (claude-sonnet-4-6) and return None gracefully
if the API key is missing, the anthropic package is not installed, or any error occurs.
"""
from __future__ import annotations

_ANTHROPIC_AVAILABLE = False
try:
    import anthropic as _anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    pass

# ── System prompts (cached at the prompt-caching tier) ────────────────────────

_EXPL_SYSTEM = (
    "You are a professional gold market analyst providing plain-English signal "
    "explanations for retail investors. Be concise (150-200 words maximum), "
    "factual, and helpful. Never give direct financial advice or recommend "
    "specific trade sizes. Always end with one key risk to watch. "
    "Use simple language — your reader is an intelligent non-expert."
)

_BRIEF_SYSTEM = (
    "You are a professional gold market analyst writing a daily morning brief "
    "for an intelligent individual investor. Write in the style of a Bloomberg "
    "brief — clear, factual, accessible. Cover the current gold price, today's "
    "AI signal, key technical levels (support/resistance from Bollinger Bands "
    "and recent price action), market regime context, and one actionable "
    "observation. End with a brief risk note. 200-250 words. "
    "This is research context only — not financial advice."
)

_MODEL = "claude-sonnet-4-6"


# ── Prompt builders ────────────────────────────────────────────────────────────

def _build_explanation_prompt(sd: dict) -> str:
    sig      = sd.get("signal",               "SIDEWAYS")
    conf_pct = sd.get("confidence", 0) * 100
    price    = sd.get("gold_price",            0.0)
    chg      = sd.get("price_change_pct",      0.0)
    rsi      = sd.get("rsi",                   50.0)
    macd     = sd.get("macd",                  0.0)
    macd_sig = sd.get("macd_signal",           0.0)
    bb_pctb  = sd.get("bb_pctb",               0.5)
    atr_pct  = sd.get("atr_pct",               1.0)
    vix      = sd.get("vix",                   20.0)
    regime   = sd.get("market_regime",         "Neutral")
    top_feat = sd.get("top_features",          [])[:3]
    dp       = sd.get("directional_probs",     {})
    days_ago = sd.get("days_since_last_signal", "N/A")
    bar_date = sd.get("last_bar_date",         "today")

    p_up = dp.get("UP",      0.0) * 100
    p_dn = dp.get("DOWN",    0.0) * 100
    p_sw = dp.get("SIDEWAYS", 0.0) * 100

    feat_str = ", ".join(top_feat) if top_feat else "N/A"
    return (
        f"Date: {bar_date}. Gold (XAU/USD) is trading at ${price:,.2f}, "
        f"{chg:+.2f}% from the previous close.\n"
        f"The AI model generated a {sig} signal with {conf_pct:.0f}% confidence.\n\n"
        f"Directional probabilities — UP: {p_up:.0f}%, SIDEWAYS: {p_sw:.0f}%, DOWN: {p_dn:.0f}%.\n"
        f"Technical readings: RSI {rsi:.1f}, MACD {macd:.4f} vs signal {macd_sig:.4f}, "
        f"Bollinger %B {bb_pctb:.2f}, ATR {atr_pct:.1f}% of price, VIX {vix:.1f}.\n"
        f"Market regime: {regime}. Top model drivers: {feat_str}.\n"
        f"Days since last directional signal: {days_ago}.\n\n"
        f"Please explain in plain English why the model generated this {sig} signal "
        f"and what the key technical conditions mean for gold market participants today."
    )


def _build_brief_prompt(sd: dict, date: str) -> str:
    sig     = sd.get("signal",             "SIDEWAYS")
    conf    = sd.get("confidence", 0) * 100
    price   = sd.get("gold_price",         0.0)
    chg     = sd.get("price_change_pct",   0.0)
    regime  = sd.get("market_regime",      "Neutral")
    atr_pct = sd.get("atr_pct",            1.0)
    bb_pctb = sd.get("bb_pctb",            0.5)
    rsi     = sd.get("rsi",                50.0)
    vix     = sd.get("vix",                20.0)
    dp      = sd.get("directional_probs",  {})
    p_up    = dp.get("UP",   0.0) * 100
    p_dn    = dp.get("DOWN", 0.0) * 100

    return (
        f"Date: {date}. Write a morning market brief for gold (XAU/USD).\n\n"
        f"Current market data:\n"
        f"- Gold price: ${price:,.2f} ({chg:+.2f}% overnight)\n"
        f"- Today's AI signal: {sig} ({conf:.0f}% model confidence)\n"
        f"- Model directional probabilities: UP {p_up:.0f}%, DOWN {p_dn:.0f}%\n"
        f"- RSI: {rsi:.1f}, Bollinger %B: {bb_pctb:.2f}, ATR: {atr_pct:.1f}% of price\n"
        f"- VIX: {vix:.1f}, Market regime: {regime}\n\n"
        f"Structure the brief as: overnight move summary, today's signal and what it "
        f"signals, key support/resistance levels (derive from BB and recent action), "
        f"regime context, one actionable observation, risk note."
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_signal_explanation(signal_data: dict, api_key: str) -> str | None:
    """
    Generate a 150-200 word plain-English explanation of today's trading signal.

    Parameters
    ----------
    signal_data : dict — see _build_explanation_prompt for expected keys
    api_key     : Anthropic API key from secrets.toml

    Returns
    -------
    explanation text (str) or None on any failure
    """
    if not _ANTHROPIC_AVAILABLE or not api_key:
        return None
    try:
        client = _anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=_MODEL,
            max_tokens=400,
            temperature=0.3,
            system=[
                {
                    "type": "text",
                    "text": _EXPL_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": _build_explanation_prompt(signal_data),
                }
            ],
        )
        return msg.content[0].text.strip()
    except Exception:
        return None


def generate_morning_brief(
    signal_data: dict,
    date: str,
    api_key: str,
) -> str | None:
    """
    Generate a 200-250 word daily morning market brief for gold.

    Parameters
    ----------
    signal_data : dict — see _build_brief_prompt for expected keys
    date        : date string (e.g. "2026-05-24")
    api_key     : Anthropic API key from secrets.toml

    Returns
    -------
    brief text (str) or None on any failure
    """
    if not _ANTHROPIC_AVAILABLE or not api_key:
        return None
    try:
        client = _anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=_MODEL,
            max_tokens=500,
            temperature=0.3,
            system=[
                {
                    "type": "text",
                    "text": _BRIEF_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": _build_brief_prompt(signal_data, date),
                }
            ],
        )
        return msg.content[0].text.strip()
    except Exception:
        return None
