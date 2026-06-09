"""
APEX Metals AI — landing page (Streamlit translation).

Self-contained, additive, presentation-only. Touches no core logic, no models,
no thresholds. All CSS is scoped under `.apex-lp` so it cannot leak into the rest
of your app.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    from apex_landing_streamlit import render_landing

    render_landing(
        signals={
            "gold":     {"verdict": "NO-TRADE", "confidence": 41.7, "regime": "Range-bound regime · no edge today"},
            "silver":   {"verdict": "NO-TRADE", "confidence": 37.8, "regime": "Low conviction · filter override"},
            "platinum": {"verdict": "CAUTION",  "confidence": 41.6, "regime": "Mixed signals · size down"},
        },
        evidence=[
            {"v": "41.1%",       "k": "Directional accuracy",         "ctx": "+20pp over a coin toss"},
            {"v": "55.6%",       "k": "DOWN-call accuracy",           "ctx": "strongest class"},
            {"v": "Walk-forward","k": "Validation method",            "ctx": "out-of-sample, not curve-fit"},
            {"v": "100%",        "k": "Calls logged & hash-chained",  "ctx": "before the outcome is known"},
        ],
        launch_url="/Dashboard",        # path to your dashboard page; "#" if unsure
        hide_streamlit_chrome=True,     # hide the default header / menu / toolbar
    )

------------------------------------------------------------------------------
WIRE POINTS
------------------------------------------------------------------------------
- signals[metal]["verdict"]    -> one of "GO" | "CAUTION" | "NO-TRADE"
- signals[metal]["confidence"] -> a number (percent), or a preformatted string
- signals[metal]["regime"]     -> a short one-line note
- evidence                     -> the honestly-framed stat strip (swap in real figures)
- launch_url                   -> where every CTA points. In multipage Streamlit use the
                                  page path (e.g. "/Dashboard"); or replace the anchors
                                  with st.page_link if you prefer native routing.

The defaults below are illustrative and match the HTML mockup; pass your live values in.

------------------------------------------------------------------------------
NOTES
------------------------------------------------------------------------------
- CSS animations (the live-dot pulse, hover transitions) work in Streamlit.
- The scroll-reveal animations from the HTML mockup are intentionally gone (they
  required JavaScript, which st.markdown strips). The page just renders fully visible.
- The sticky nav relies on position:sticky; if it behaves oddly with your Streamlit
  version, change `.apex-lp header.nav { position: sticky }` to `position: static`.
"""

import streamlit as st

# --------------------------------------------------------------------------- #
# Defaults (illustrative — pass your live values into render_landing)
# --------------------------------------------------------------------------- #
DEFAULT_SIGNALS = {
    "gold":     {"verdict": "NO-TRADE", "confidence": 41.7, "regime": "Range-bound regime · no edge today"},
    "silver":   {"verdict": "NO-TRADE", "confidence": 37.8, "regime": "Low conviction · filter override"},
    "platinum": {"verdict": "CAUTION",  "confidence": 41.6, "regime": "Mixed signals · size down"},
}

DEFAULT_EVIDENCE = [
    {"v": "41.1%",        "k": "Directional accuracy",        "ctx": "+20pp over a coin toss"},
    {"v": "55.6%",        "k": "DOWN-call accuracy",          "ctx": "strongest class"},
    {"v": "Walk-forward", "k": "Validation method",           "ctx": "out-of-sample, not curve-fit"},
    {"v": "100%",         "k": "Calls logged & hash-chained", "ctx": "before the outcome is known"},
]

_METALS = [
    ("gold",     "Gold",     "XAU / USD", "Au", "au"),
    ("silver",   "Silver",   "XAG / USD", "Ag", "ag"),
    ("platinum", "Platinum", "XPT / USD", "Pt", "pt"),
]

_VERDICT = {
    "GO":       ("v-go",      "Go"),
    "CAUTION":  ("v-caution", "Caution"),
    "NO-TRADE": ("v-no",      "No-trade"),
    "NO_TRADE": ("v-no",      "No-trade"),
    "NOTRADE":  ("v-no",      "No-trade"),
}


def _verdict_meta(v):
    return _VERDICT.get(str(v).upper().strip(), ("v-no", str(v)))


def _fmt_conf(c):
    try:
        return f"{float(c):.1f}%"
    except (TypeError, ValueError):
        return str(c)


# --------------------------------------------------------------------------- #
# Styles (scoped under .apex-lp so nothing leaks into the rest of the app)
# --------------------------------------------------------------------------- #
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root{
  --ink:#0A0D14; --ink-2:#0C1018; --raised:#121826; --raised-2:#161D2C;
  --line:rgba(255,255,255,.07); --line-gold:rgba(232,179,65,.20);
  --gold:#E8B341; --gold-bright:#F2CD6E; --gold-deep:#9A6F22;
  --silver:#AEB7C4; --platinum:#A78BFA;
  --go:#41C463; --caution:#E0913A; --neutral:#7C8696;
  --text:#E9ECF2; --muted:#8B93A3; --dim:#5A6273;
  --sans:'Inter',system-ui,sans-serif; --disp:'Space Grotesk',var(--sans); --mono:'IBM Plex Mono',monospace;
  --r:14px; --maxw:1080px;
}
html{scroll-behavior:smooth}

/* let the landing page go full-bleed inside Streamlit's container */
.stApp,[data-testid="stAppViewContainer"]{background:#0A0D14}
[data-testid="stMainBlockContainer"],.block-container{padding:0 !important;max-width:100% !important}

.apex-lp{color:var(--text);font-family:var(--sans);font-size:16px;line-height:1.65;-webkit-font-smoothing:antialiased}
.apex-lp *{box-sizing:border-box;margin:0;padding:0}
.apex-lp a{color:inherit;text-decoration:none}
.apex-lp ::selection{background:rgba(232,179,65,.25)}
.apex-lp :focus-visible{outline:2px solid var(--gold);outline-offset:3px;border-radius:4px}
.apex-lp .wrap{max-width:var(--maxw);margin:0 auto;padding:0 28px}
.apex-lp .mono{font-family:var(--mono)}
.apex-lp .eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:var(--dim)}

.apex-lp header.nav{position:sticky;top:0;z-index:50;backdrop-filter:blur(10px);background:rgba(10,13,20,.72);border-bottom:1px solid var(--line)}
.apex-lp .nav-in{display:flex;align-items:center;justify-content:space-between;height:64px}
.apex-lp .brand{display:flex;align-items:center;gap:10px;font-family:var(--disp);font-weight:700;font-size:17px;letter-spacing:.01em;color:var(--text)}
.apex-lp .brand .dia{width:14px;height:14px;transform:rotate(45deg);background:var(--gold);border-radius:3px;box-shadow:0 0 0 4px rgba(232,179,65,.12)}
.apex-lp .nav-links{display:flex;align-items:center;gap:28px}
.apex-lp .nav-links a{font-size:14px;color:var(--muted);transition:color .2s}
.apex-lp .nav-links a:hover{color:var(--text)}
.apex-lp .btn{display:inline-flex;align-items:center;gap:8px;font-weight:600;font-size:14px;border-radius:10px;padding:10px 18px;cursor:pointer;border:1px solid transparent;transition:transform .15s,background .2s,border-color .2s;white-space:nowrap}
.apex-lp .btn-gold{background:var(--gold);color:#1A1305}
.apex-lp .btn-gold:hover{background:var(--gold-bright);transform:translateY(-1px)}
.apex-lp .btn-ghost{background:transparent;border-color:var(--line);color:var(--text)}
.apex-lp .btn-ghost:hover{border-color:var(--line-gold);background:rgba(232,179,65,.05)}
.apex-lp .nav .btn{padding:8px 16px}

.apex-lp .hero{position:relative;text-align:center;padding:84px 0 64px}
.apex-lp .hero::before{content:"";position:absolute;inset:0;background:radial-gradient(620px 320px at 50% -40px,rgba(232,179,65,.10),transparent 70%);pointer-events:none}
.apex-lp .pills{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-bottom:34px;position:relative}
.apex-lp .pill{display:inline-flex;align-items:center;gap:8px;font-family:var(--mono);font-size:12.5px;letter-spacing:.04em;padding:7px 15px;border-radius:999px;border:1px solid var(--line);color:var(--muted)}
.apex-lp .pill .dot{width:7px;height:7px;border-radius:50%}
.apex-lp .pill.au{border-color:var(--line-gold)} .apex-lp .pill.au .dot{background:var(--gold)} .apex-lp .pill.au b{color:var(--gold)}
.apex-lp .pill.ag .dot{background:var(--silver)} .apex-lp .pill.ag b{color:var(--silver)}
.apex-lp .pill.pt{border-color:rgba(167,139,250,.3)} .apex-lp .pill.pt .dot{background:var(--platinum)} .apex-lp .pill.pt b{color:var(--platinum)}
.apex-lp .pill b{font-weight:500}
.apex-lp h1.brandbig{font-family:var(--disp);font-weight:700;font-size:clamp(38px,7vw,72px);line-height:1;letter-spacing:-.02em;color:var(--gold);position:relative;margin-bottom:22px}
.apex-lp .thesis{font-family:var(--disp);font-weight:500;font-size:clamp(20px,3.4vw,30px);line-height:1.25;letter-spacing:-.01em;max-width:18ch;margin:0 auto 22px;position:relative}
.apex-lp .thesis .nt{color:var(--gold)}
.apex-lp .lede{max-width:60ch;margin:0 auto;color:var(--muted);font-size:clamp(15px,1.6vw,17px);position:relative}
.apex-lp .cta-row{display:flex;gap:12px;justify-content:center;margin:34px 0 18px;position:relative;flex-wrap:wrap}
.apex-lp .disc{font-family:var(--mono);font-size:12px;color:var(--dim);letter-spacing:.02em;position:relative}

.apex-lp section{padding:72px 0}
.apex-lp .sec-head{margin-bottom:36px}
.apex-lp .sec-head .eyebrow{display:block;margin-bottom:12px}
.apex-lp .sec-head h2{font-family:var(--disp);font-weight:500;font-size:clamp(24px,3.4vw,34px);letter-spacing:-.01em;line-height:1.15;color:var(--text)}
.apex-lp .sec-head p{color:var(--muted);margin-top:12px;max-width:62ch}
.apex-lp .center{text-align:center} .apex-lp .center .sec-head p{margin-left:auto;margin-right:auto}

.apex-lp .signals{background:var(--ink-2);border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
.apex-lp .cards-3{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}
.apex-lp .card{background:var(--raised);border:1px solid var(--line);border-radius:var(--r);padding:22px 22px 18px;position:relative;transition:transform .2s,border-color .2s}
.apex-lp .card:hover{transform:translateY(-3px)}
.apex-lp .card .accent{position:absolute;top:0;left:22px;right:22px;height:2px;border-radius:2px;opacity:.85}
.apex-lp .card.au .accent{background:var(--gold)} .apex-lp .card.au:hover{border-color:var(--line-gold)}
.apex-lp .card.ag .accent{background:var(--silver)} .apex-lp .card.ag:hover{border-color:rgba(174,183,196,.3)}
.apex-lp .card.pt .accent{background:var(--platinum)} .apex-lp .card.pt:hover{border-color:rgba(167,139,250,.3)}
.apex-lp .card-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px}
.apex-lp .metal{display:flex;align-items:center;gap:10px}
.apex-lp .glyph{font-family:var(--mono);font-weight:500;font-size:13px;width:30px;height:30px;display:grid;place-items:center;border-radius:8px;border:1px solid var(--line)}
.apex-lp .card.au .glyph{color:var(--gold)} .apex-lp .card.ag .glyph{color:var(--silver)} .apex-lp .card.pt .glyph{color:var(--platinum)}
.apex-lp .metal .nm{font-weight:600;font-size:15px} .apex-lp .metal .tk{font-family:var(--mono);font-size:11px;color:var(--dim)}
.apex-lp .verdict{font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;padding:4px 10px;border-radius:7px;border:1px solid}
.apex-lp .v-go{color:var(--go);border-color:rgba(65,196,99,.35);background:rgba(65,196,99,.08)}
.apex-lp .v-caution{color:var(--caution);border-color:rgba(224,145,58,.35);background:rgba(224,145,58,.08)}
.apex-lp .v-no{color:var(--neutral);border-color:var(--line);background:rgba(255,255,255,.03)}
.apex-lp .conf{display:flex;align-items:baseline;gap:7px}
.apex-lp .conf .num{font-family:var(--mono);font-weight:500;font-size:30px;letter-spacing:-.01em}
.apex-lp .conf .lab{font-size:12px;color:var(--dim)}
.apex-lp .regime{color:var(--muted);font-size:13px;margin-top:6px}
.apex-lp .card-foot{margin-top:16px;padding-top:12px;border-top:1px solid var(--line);font-family:var(--mono);font-size:11px;color:var(--dim);display:flex;align-items:center;gap:7px}
.apex-lp .livedot{width:6px;height:6px;border-radius:50%;background:var(--go);animation:apexpulse 2.4s infinite}
@keyframes apexpulse{0%{box-shadow:0 0 0 0 rgba(65,196,99,.45)}70%{box-shadow:0 0 0 6px rgba(65,196,99,0)}100%{box-shadow:0 0 0 0 rgba(65,196,99,0)}}

.apex-lp .chain-band{background:linear-gradient(180deg,var(--ink),var(--ink-2));position:relative}
.apex-lp .chain-inner{border:1px solid var(--line-gold);border-radius:18px;padding:40px clamp(22px,4vw,48px);background:rgba(18,24,38,.5)}
.apex-lp .chain-copy h2{font-family:var(--disp);font-weight:500;font-size:clamp(23px,3.2vw,32px);line-height:1.18;letter-spacing:-.01em;margin-bottom:14px;color:var(--text)}
.apex-lp .chain-copy h2 .br{color:var(--gold)}
.apex-lp .chain-copy p{color:var(--muted);max-width:60ch}
.apex-lp .chain-tags{display:flex;gap:10px;flex-wrap:wrap;margin-top:20px}
.apex-lp .tag{font-family:var(--mono);font-size:12px;color:var(--muted);border:1px solid var(--line);border-radius:8px;padding:7px 12px;display:inline-flex;align-items:center;gap:8px}
.apex-lp .tag svg{width:14px;height:14px;color:var(--gold)}
.apex-lp .chain-viz{display:flex;align-items:center;gap:0;overflow-x:auto;padding:24px 2px 6px}
.apex-lp .node{flex:0 0 auto;display:flex;flex-direction:column;align-items:center;gap:8px}
.apex-lp .node .hashchip{font-family:var(--mono);font-size:12px;color:var(--muted);background:var(--raised-2);border:1px solid var(--line);border-radius:9px;padding:10px 13px;white-space:nowrap}
.apex-lp .node.head .hashchip{color:var(--gold);border-color:var(--line-gold);background:rgba(232,179,65,.07)}
.apex-lp .node .lbl{font-family:var(--mono);font-size:10px;color:var(--dim);letter-spacing:.04em}
.apex-lp .link{flex:0 0 auto;width:36px;height:1px;background:var(--line-gold);position:relative;margin-top:-14px}
.apex-lp .link::after{content:"";position:absolute;right:-1px;top:-2px;width:5px;height:5px;border-top:1px solid var(--gold-deep);border-right:1px solid var(--gold-deep);transform:rotate(45deg)}
.apex-lp .verified{flex:0 0 auto;display:flex;flex-direction:column;align-items:center;gap:8px;margin-left:6px}
.apex-lp .verified .vc{width:34px;height:34px;border-radius:50%;display:grid;place-items:center;border:1px solid rgba(65,196,99,.4);background:rgba(65,196,99,.08);color:var(--go)}
.apex-lp .verified .lbl{font-family:var(--mono);font-size:10px;color:var(--go);letter-spacing:.04em}

.apex-lp .grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}
.apex-lp .trust-card{padding:24px 22px;border:1px solid var(--line);border-radius:var(--r);background:var(--raised)}
.apex-lp .trust-card .ic{width:38px;height:38px;border-radius:10px;display:grid;place-items:center;border:1px solid var(--line-gold);color:var(--gold);margin-bottom:16px}
.apex-lp .trust-card .ic svg{width:20px;height:20px}
.apex-lp .trust-card h3{font-family:var(--disp);font-weight:500;font-size:17px;margin-bottom:8px;color:var(--text)}
.apex-lp .trust-card p{color:var(--muted);font-size:14px}

.apex-lp .evidence{background:var(--ink-2);border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
.apex-lp .stat-row{display:grid;grid-template-columns:repeat(4,1fr);gap:18px}
.apex-lp .stat{padding:22px;border-radius:var(--r);background:var(--raised);border:1px solid var(--line)}
.apex-lp .stat .v{font-family:var(--mono);font-weight:500;font-size:clamp(24px,3vw,30px);color:var(--gold);letter-spacing:-.01em}
.apex-lp .stat .k{font-size:13px;color:var(--muted);margin-top:6px}
.apex-lp .stat .ctx{font-family:var(--mono);font-size:11px;color:var(--dim);margin-top:8px}
.apex-lp .fineprint{color:var(--dim);font-size:13px;margin-top:22px;max-width:70ch}

.apex-lp .feat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}
.apex-lp .feat{padding:24px 22px;border:1px solid var(--line);border-radius:var(--r);background:var(--raised);transition:border-color .2s}
.apex-lp .feat:hover{border-color:var(--line-gold)}
.apex-lp .feat .ic{color:var(--gold);margin-bottom:16px}
.apex-lp .feat .ic svg{width:24px;height:24px}
.apex-lp .feat h3{font-family:var(--disp);font-weight:500;font-size:16px;margin-bottom:8px;color:var(--text)}
.apex-lp .feat p{color:var(--muted);font-size:13.5px}

.apex-lp .final{text-align:center;padding:84px 0}
.apex-lp .final h2{font-family:var(--disp);font-weight:500;font-size:clamp(26px,4vw,40px);letter-spacing:-.01em;line-height:1.12;margin-bottom:16px;color:var(--text)}
.apex-lp .final p{color:var(--muted);max-width:52ch;margin:0 auto 30px}

.apex-lp footer{border-top:1px solid var(--line);padding:48px 0 56px;background:var(--ink-2)}
.apex-lp .foot-top{display:flex;justify-content:space-between;gap:24px;flex-wrap:wrap;margin-bottom:28px}
.apex-lp .foot-links{display:flex;gap:26px;flex-wrap:wrap}
.apex-lp .foot-links a{font-size:14px;color:var(--muted)} .apex-lp .foot-links a:hover{color:var(--text)}
.apex-lp .legal{font-family:var(--mono);font-size:12px;color:var(--dim);line-height:1.8;border-top:1px solid var(--line);padding-top:22px}

@media (max-width:820px){
  .apex-lp .cards-3,.apex-lp .grid-3,.apex-lp .feat-grid{grid-template-columns:1fr}
  .apex-lp .stat-row{grid-template-columns:1fr 1fr}
  .apex-lp section{padding:56px 0}
}
@media (max-width:520px){ .apex-lp .stat-row{grid-template-columns:1fr} }
@media (prefers-reduced-motion:reduce){ .apex-lp .livedot{animation:none} html{scroll-behavior:auto} }
"""

_CHROME_HIDE_CSS = """
[data-testid="stHeader"]{display:none}
[data-testid="stToolbar"]{display:none}
#MainMenu{display:none}
[data-testid="stSidebar"]{display:none !important}
[data-testid="stSidebarNav"]{display:none !important}
"""

# small, consistent line-icons
_IC = {
    "shield": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6z"/><path d="M9 12l2 2 4-4"/></svg>',
    "clock":  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
    "check":  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M5 12l5 5 9-11"/></svg>',
    "pulse":  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M4 12h4l2-6 4 12 2-6h4"/></svg>',
    "links":  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M9 12a3 3 0 013-3h3a3 3 0 010 6h-1"/><path d="M15 12a3 3 0 01-3 3H9a3 3 0 010-6h1"/></svg>',
    "chart":  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M3 17l5-6 4 4 5-7 4 5"/></svg>',
    "chip":   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><rect x="5" y="5" width="14" height="14" rx="2"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/></svg>',
    "sun":    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><circle cx="12" cy="12" r="4"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2"/></svg>',
    "gate":   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M12 4l8 4v5c0 4-3 7-8 8-5-1-8-4-8-8V8z"/></svg>',
    "bell":   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M18 8a6 6 0 10-12 0c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M10 21a2 2 0 004 0"/></svg>',
    "globe":  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3 3 15 0 18M12 3c-3 3-3 15 0 18"/></svg>',
}


def _signal_card(key, label, ticker, glyph, accent, data):
    vclass, vlabel = _verdict_meta(data.get("verdict", "NO-TRADE"))
    conf = _fmt_conf(data.get("confidence", ""))
    regime = data.get("regime", "")
    return f"""
      <div class="card {accent}"><span class="accent"></span>
        <div class="card-top"><div class="metal"><span class="glyph">{glyph}</span><div><div class="nm">{label}</div><div class="tk">{ticker}</div></div></div><span class="verdict {vclass}">{vlabel}</span></div>
        <div class="conf"><span class="num">{conf}</span><span class="lab">model confidence</span></div>
        <div class="regime">{regime}</div>
        <div class="card-foot"><span class="livedot"></span> Live · as of last settled close</div>
      </div>"""


def render_landing(*, signals=None, evidence=None, launch_url="#", hide_streamlit_chrome=True):
    """Render the APEX Metals AI landing page. Presentation-only and additive."""
    signals = {**DEFAULT_SIGNALS, **(signals or {})}
    evidence = evidence or DEFAULT_EVIDENCE

    css = _CSS + (_CHROME_HIDE_CSS if hide_streamlit_chrome else "")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    cards = "".join(
        _signal_card(k, label, tk, gl, ac, signals.get(k, {}))
        for (k, label, tk, gl, ac) in _METALS
    )

    stats = "".join(
        f'<div class="stat"><div class="v">{s["v"]}</div><div class="k">{s["k"]}</div><div class="ctx">{s["ctx"]}</div></div>'
        for s in evidence
    )

    html = f"""
<div class="apex-lp">
  <header class="nav">
    <div class="wrap nav-in">
      <a class="brand" href="#top"><span class="dia"></span> APEX Metals AI</a>
      <nav class="nav-links" aria-label="Primary">
        <a href="#signals">Signals</a>
        <a href="#record">Track record</a>
        <a href="#inside">What's inside</a>
        <a class="btn btn-gold" href="{launch_url}">Launch dashboard</a>
      </nav>
    </div>
  </header>

  <main id="top">
    <section class="hero">
      <div class="wrap">
        <div class="pills">
          <span class="pill au"><span class="dot"></span><b>Gold</b> · XAU</span>
          <span class="pill ag"><span class="dot"></span><b>Silver</b> · XAG</span>
          <span class="pill pt"><span class="dot"></span><b>Platinum</b> · XPT</span>
        </div>
        <h1 class="brandbig">APEX Metals AI</h1>
        <p class="thesis">Decision intelligence for precious metals — including the decision <span class="nt">not to trade</span>.</p>
        <p class="lede">An ensemble ML signal engine for gold, silver, and platinum — regime-aware, calibrated, and uncertainty-forward. Every prediction is logged before the outcome is known, hash-chained, and independently verifiable.</p>
        <div class="cta-row">
          <a class="btn btn-gold" href="{launch_url}">Launch dashboard</a>
          <a class="btn btn-ghost" href="#inside">How it works</a>
        </div>
        <p class="disc">Personal research only · Not financial advice · Past performance ≠ future results</p>
      </div>
    </section>

    <section class="signals" id="signals">
      <div class="wrap">
        <div class="sec-head">
          <span class="eyebrow">Today's read</span>
          <h2>Where the models stand right now</h2>
          <p>A live directional read per metal, gated by the decision-intelligence layer. When conviction isn't there, it says so.</p>
        </div>
        <div class="cards-3">{cards}</div>
      </div>
    </section>

    <section class="chain-band" id="record">
      <div class="wrap">
        <div class="chain-inner">
          <span class="eyebrow" style="display:block;margin-bottom:14px">Verifiable by design</span>
          <div class="chain-copy">
            <h2>A track record that <span class="br">can't be quietly rewritten</span>.</h2>
            <p>Every signal is recorded the moment it's made — before the outcome is known — and linked to the one before it with a cryptographic hash. Change any past record and the chain breaks. The history is auditable, not editable.</p>
            <div class="chain-tags">
              <span class="tag">{_IC['clock']} Logged before the outcome</span>
              <span class="tag">{_IC['shield']} Tamper-evident chain</span>
              <span class="tag">{_IC['check']} Settled-bar discipline</span>
            </div>
          </div>
          <div class="chain-viz" role="img" aria-label="A chain of hash-linked prediction records ending in a verified marker">
            <div class="node"><span class="hashchip">2c1a…f704</span><span class="lbl">genesis</span></div>
            <div class="link"></div>
            <div class="node"><span class="hashchip">9f07…b3aa</span><span class="lbl">record 2</span></div>
            <div class="link"></div>
            <div class="node"><span class="hashchip">ae0b…d1e6</span><span class="lbl">record 3</span></div>
            <div class="link"></div>
            <div class="node head"><span class="hashchip">4736…0a85</span><span class="lbl">latest</span></div>
            <div class="verified"><span class="vc">{_IC['check']}</span><span class="lbl">chain verified</span></div>
          </div>
        </div>
      </div>
    </section>

    <section>
      <div class="wrap">
        <div class="sec-head center">
          <span class="eyebrow">Built to be trusted</span>
          <h2>Honest by construction</h2>
          <p>The category is full of tools that only know how to say buy. APEX is built around the opposite instinct.</p>
        </div>
        <div class="grid-3">
          <div class="trust-card"><div class="ic">{_IC['shield']}</div><h3>Earned GO verdicts</h3><p>A GO is issued only on positive evidence, gated by the decision layer — never as a default. Most calls are deliberately not GO.</p></div>
          <div class="trust-card"><div class="ic">{_IC['pulse']}</div><h3>Uncertainty-forward</h3><p>It shows caution and no-trade plainly. When the signal isn't there, it tells you — instead of manufacturing one to look useful.</p></div>
          <div class="trust-card"><div class="ic">{_IC['links']}</div><h3>Independently verifiable</h3><p>The hash-chained record means the track record can be checked, not just claimed. Trust is a property of the data, not a promise.</p></div>
        </div>
      </div>
    </section>

    <section class="evidence">
      <div class="wrap">
        <div class="sec-head">
          <span class="eyebrow">Measured honestly</span>
          <h2>The metrics that matter — with context</h2>
          <p>Directional forecasting is hard, and we report it straight rather than behind a single flattering number.</p>
        </div>
        <div class="stat-row">{stats}</div>
        <p class="fineprint">Three-way UP / SIDEWAYS / DOWN forecasting against a ~33% random baseline. Headline accuracy on its own is a poor summary of a directional model, so we lead with the figures that actually carry information — and let the live, verifiable record speak for itself over time.</p>
      </div>
    </section>

    <section id="inside">
      <div class="wrap">
        <div class="sec-head center">
          <span class="eyebrow">What's inside</span>
          <h2>A research instrument, not a tip sheet</h2>
        </div>
        <div class="feat-grid">
          <div class="feat"><div class="ic">{_IC['chart']}</div><h3>Resilient live pricing</h3><p>A four-tier price feed with LBMA and COMEX benchmarks, so a single source outage never blinds the read.</p></div>
          <div class="feat"><div class="ic">{_IC['chip']}</div><h3>Regime-aware signals</h3><p>An ensemble that adapts to the prevailing market regime, tuned and validated walk-forward.</p></div>
          <div class="feat"><div class="ic">{_IC['sun']}</div><h3>Daily AI brief</h3><p>A plain-English read on the precious-metals session each morning, generated by Claude.</p></div>
          <div class="feat"><div class="ic">{_IC['gate']}</div><h3>Decision intelligence</h3><p>The GO / CAUTION / NO-TRADE gate that turns a raw probability into an actual, accountable call.</p></div>
          <div class="feat"><div class="ic">{_IC['bell']}</div><h3>Risk-aware alerts</h3><p>Signal and risk alerts by email, rate-limited so the ones that arrive still mean something.</p></div>
          <div class="feat"><div class="ic">{_IC['globe']}</div><h3>Multi-currency view</h3><p>Prices and signals across nine currencies, including AED and USD, for a local read.</p></div>
        </div>
      </div>
    </section>

    <section class="final" id="launch">
      <div class="wrap">
        <h2>See where the models stand today.</h2>
        <p>Open the dashboard for the live read across gold, silver, and platinum — and the verifiable record behind it.</p>
        <a class="btn btn-gold" href="{launch_url}" style="padding:13px 26px;font-size:15px">Launch dashboard</a>
      </div>
    </section>
  </main>

  <footer>
    <div class="wrap">
      <div class="foot-top">
        <a class="brand" href="#top"><span class="dia"></span> APEX Metals AI</a>
        <nav class="foot-links" aria-label="Footer">
          <a href="#signals">Signals</a>
          <a href="#record">Track record</a>
          <a href="#inside">What's inside</a>
        </nav>
      </div>
      <p class="legal">APEX Metals AI is a personal research tool. Nothing here is financial advice. Market data is delayed. Past performance is not indicative of future results.<br>© 2026 APEX Metals AI.</p>
    </div>
  </footer>
</div>
"""
    html = "".join(line.strip() for line in html.splitlines())
    st.markdown(html, unsafe_allow_html=True)


if __name__ == "__main__":
    # Local preview: `streamlit run apex_landing_streamlit.py`
    st.set_page_config(page_title="APEX Metals AI", layout="wide")
    render_landing()
