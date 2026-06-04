# APEX Metals AI — Phase 6 Build State

## Current Status (as of 2026-06-04)

### Steps Completed

**Step 1 — Platform Rename: COMPLETE**
- All "Gold AI Decision Intelligence" / "Gold AI" text replaced with "APEX Metals AI" across:
  `app.py`, `src/alerts.py`, `pages/2_Risk_Management.py`,
  `pages/3_Historical_Performance.py`, `pages/4_Live_Validation.py`,
  `README.md`, `docs/BUG_FIX_LOG.md`, `screenshot_dash.py`, `screenshot_opt.py`
- `st.set_page_config(title="APEX Metals AI", page_icon="🏅")`
- `XAU/USD Morning Brief` preserved in `src/explainer.py` (market term)
- Python variable/function names, file names, folder structure unchanged
- Not committed yet (per instructions)

**Step 2 — Landing Page: COMPLETE**
- `show_landing_page()` function added to `app.py`
- Session state gate: `if not st.session_state.show_dashboard: show_landing_page(); st.stop()`
- Landing page sections: Hero (metal pills, gold-gradient title, tagline, Launch button),
  Stats bar (4 cols), Metals cards (3 cols), Features grid (2×3), Model performance
  metrics, Roadmap (Phases 1–6), Footer
- Dark theme CSS (#070f18 bg, #b8960c gold accents), sidebar hidden on landing page
- "← Back to overview" button at top of dashboard (`key="back_btn"`)
- Not committed yet

**Step 3 — Metals Navigation: COMPLETE**
- `st.radio` metal selector (Gold / Silver / Platinum) at top of dashboard
- Silver and Platinum show live price (yfinance SI=F / PL=F) + "Coming in Phase 6B" info panel
- `st.stop()` after Silver/Platinum placeholder so Gold dashboard code runs unchanged
- `st.session_state.selected_metal` persists selection
- Not committed yet

**Step 4 — Model Accuracy Upgrade: COMPLETE & LOCKED**

*4A – Macro feature layer*: `fetch_macro_data()` added to `src/features.py`.
New features using existing `_Close` columns (DXY, TNX, OIL, VIX, SPY):
`{name}_close`, `{name}_chg1d`, `{name}_chg5d`, `{name}_vs_sma20` for all 5 series.
Cross-asset: `gold_dxy_corr20`, `gold_oil_corr20`, `real_rate_proxy`, `risk_off_index`.

*4B – Seasonality*: `quarter`, `day_of_week`, `week_of_year`, `q4_demand`,
`january_effect`, `summer_low_vol`, `is_monday`, `is_friday` added to `add_features()`.

*4C – Regime classification*: `classify_regime(df, window=20)` added to `src/features.py`.
Labels: `high_vol`, `trending`, `neutral`. Column `regime` excluded from model features
via `get_feature_columns()` (string dtype guard added).

*4D – Regime-conditional models*: `train_regime_models()` added to `src/train.py`.
Three LightGBM models (one per regime) trained after main ensemble.
`sideways_weight_boost` parameter added to `train_all_models()` (default 1.30,
set to 1.0 for v6.0 retrain to fix DOWN recall collapse).

*4E – Feature drift detection*: `detect_feature_drift()` added to `src/features.py`.
Drift score displayed in Data Sources & Integrity panel in `app.py` (green/amber/red).
`compute_training_stats()` added to `src/train.py`; saves `models/training_stats.json`.

*4F – Retrain: COMPLETE & ACCEPTED*

Full Optuna retrain (N_TRIALS=100, 18.4 min):
- `sideways_weight_boost=1.0` (SIDEWAYS tilt removed — was root cause of DOWN recall collapse)
- Extended threshold grid: DOWN floor ≥ 25% recall enforced
- **Final thresholds: DOWN > 0.30 / UP > 0.40**

| Class    | Precision | Recall | F1    |
|----------|-----------|--------|-------|
| DOWN     | 36.2%     | 30.0%  | 0.328 |
| SIDEWAYS | 36.9%     | 43.2%  | 0.398 |
| UP       | 40.0%     | 38.7%  | 0.393 |
| **Overall** | | **37.8%** | |

- 146 features (28 new vs v5)
- Regime models: high_vol (178 samples), trending (344), neutral (544) — all trained
- Training stats: 146 features saved to `models/training_stats.json`
- Model saved to `models/models.pkl`
- Retrain scripts: `run_retrain_v6.py` (fast), `run_retrain_v6_full.py` (full Optuna)

**Step 5 — Decision Intelligence Centre: COMPLETE**
- `compute_decision_verdict()` and `compute_trade_zones()` added as module-level helpers
- Section renders after Signal Strength, Gold-only gate (`selected_metal == "🥇 Gold (XAU)"`)
- 5A: GO/CAUTION/NO-TRADE verdict card with supporting/against reason columns
- 5B: ATR Research Zones 4-col metric row (Entry/Target/Stop/R:R), hidden for SIDEWAYS
- 5C: XGBoost/LightGBM/CatBoost consensus pills + unanimous/majority/split badge
- Verdict gate: GO requires against≤1 AND for≥2; otherwise CAUTION (not defaulted)
- Rolling accuracy source-tagged: "· OOS eval" or "· test set" in reason text

**Step 6A — Silver & Platinum Price Data: COMPLETE**

What ran:
- `get_silver_price()`, `get_platinum_price()`, `download_metal_ohlcv()` added to `src/data_loader.py`
- Silver waterfall: SI=F → SLV×10 → Alpha Vantage XAG ✓
- Platinum waterfall: PL=F → PPLT ✓
- `download_metal_ohlcv()` downloads OHLCV + macro sidecars with exact column names for `add_features()` ✓
- Silver/Platinum price header: live price in selected currency, **daily change $ and %** (vs prev close),
  data source label, Gold/Silver ratio (with >80 / <50 flags), Platinum/Gold spread ✓
- Metal model training buttons added to sidebar; training flow wired ✓
- OHLCV data loaded once per page render and reused across chart, delta, and DIC features ✓

**Step 6B — Silver & Platinum Signal Models: COMPLETE — HONEST DIRECTIONAL MODELS**

What ran (final state, 2026-06-04):
- `train_metal_model()`, `save_metal_models()`, `load_metal_models()` added to `src/train.py`
- Both models trained (no Optuna, fast_retrain=True); saved to `models/metals_models.pkl`
- 150 features each; LSTM augmented meta-clf active; regime-conditional models trained
- Thresholds tuned by OOF calibration:
  Silver UP>0.40/DN>0.27, Platinum UP>0.44/DN>0.33

Bugs fixed before final retrain (all in this session):
1. **classify_regime() fallback fixed** — `train_metal_model()` previously stripped OHLCV
   columns before passing to `train_all_models()`, so `classify_regime()` fell back to
   "neutral" for every row. Fix: preserve full df width; drop NaN only on training-critical
   columns. Regime counts now non-trivial for both metals.
2. **LGB sample-weight bug fixed** — `make_lgb_clf_factory()` (ensemble.py) and
   `_eval_individual_classifiers()` (train.py) both hardcoded `class_weight="balanced"`,
   bypassing the sideways_weight_boost sample weights. Fixed: class_weight removed; LGB
   now receives the same boosted sample weights as XGB/CB.
3. **SMOTE drops sample weights fixed** — `StackingClassifier.fit()` was setting
   `sw_fit=None` whenever SMOTE fired, discarding the sideways boost. Fixed: recompute
   sample weights from `cw_dict` on SMOTE-resampled labels (mirrors existing logic in
   `_eval_individual_classifiers`). SMOTE leakage audit: clean — SMOTE is only ever
   called on training-fold data, never crossing the 80/20 temporal split.

Gold is NOT affected by these fixes: gold's training path passes the full df (including
Close and regime) to `train_all_models()` and does not need a retrain.

SIDEWAYS investigation outcome — 20% recall floor NOT met (user confirmed: proceed):
- Per-metal boost grid search [1.0 … 5.0] run. Silver already meets probe threshold at
  boost=1.0 (single-LGB 25.6% SIDE recall) but the stacking meta-learner suppresses it.
- Platinum SIDEWAYS is structurally not separable with current 150 features (probe max 6.7%).
- Boosting SIDEWAYS heavily (e.g. boost=5.0 for Platinum) collapsed DOWN recall to 24.0%.
- Decision: both metals use boost=1.0 (honest directional models; no artificial SIDEWAYS).

Per-class metrics on strictly-forward 80/20 test split (no leakage, no SMOTE on test):

| Model    | Overall | DOWN rec | DOWN pre | SIDE rec | SIDE pre | UP rec | UP pre | Boost |
|----------|---------|----------|----------|----------|----------|--------|--------|-------|
| Silver   | 49.8%   | 44.3%    | 49.2%    | 10.3%    | 21.1%    | 67.9%  | 54.1%  | 1.0   |
| Platinum | 37.6%   | 42.7%    | 30.2%    | 6.7%     | 13.3%    | 42.6%  | 50.0%  | 1.0   |

3-class random baseline = 33.3%. Silver exceeds baseline significantly (+16.5pp).
Platinum marginally above baseline (+4.3pp), with strong directional recall but weak precision.

SIDEWAYS status (locked — do not retry boosting or threshold-force):
- Silver SIDEWAYS: learnable by a single LGB (probe 25.6%) but suppressed by the stacking
  meta-learner, which optimises for overall OOF accuracy and discounts a class that is
  uncertain across all three L1 models. Not a data problem; an architecture tradeoff.
- Platinum SIDEWAYS: structurally not separable with current features. Probe never exceeded
  6.7% at any boost. Accept as a directional-only model.
- In Step 6C: SIDEWAYS signals for both metals will be flagged low-reliability in the UI.
  The no-trade / confidence / consensus checks in the Decision Intelligence Centre will gate
  choppy days rather than relying on the SIDEWAYS class label.

**Step 6C — Full Silver & Platinum Dashboard: COMPLETE — awaiting review**

All sections built in `app.py`, replacing the old placeholder+`st.stop()`:
- Price header with daily change $/% in selected currency ✓
- Signal card + confidence + regime pill (inline below confidence) ✓
- SIDEWAYS low-reliability warning banner (Silver and Platinum, different copy per metal) ✓
- Directional Probability breakdown (DOWN/SIDEWAYS/UP) ✓
- Signal Strength bar (same as gold) ✓
- Morning Brief: `generate_morning_brief()` updated with `metal_name`/`metal_symbol` params;
  brief text references the correct metal; cached per-metal in session state ✓
- 90-day OHLC chart (reuses single `_load_metal_data()` call) ✓
- Decision Intelligence Centre:
  - Same GO/CAUTION/NO-TRADE gate (against≥3 → NO-TRADE; against≥2 → CAUTION;
    against≤1 AND for≥2 → GO; else → CAUTION). NOT green-by-default.
  - Each metal uses its own model and thresholds from `metals_models.pkl`
  - SIDEWAYS triggers additional low-reliability warning inside DIC
  - ATR Research Zones (UP/DOWN only) ✓
  - Classifier Consensus (XGB/LGB/CB votes from metal bundle) ✓
- Feature Importance (Top 15, bar chart in metal accent colour) ✓
- AI Signal Explanation (cached per-metal in session state) ✓
- Model status footer with per-class recall/precision and SIDEWAYS reliability note ✓

`src/explainer.py`: `_build_brief_prompt()` and `generate_morning_brief()` now accept
`metal_name` and `metal_symbol` keyword args (default "Gold"/"XAU/USD") — backward-compatible.

**Step 6D — Multi-Metal Comparison Panel: NOT STARTED**
Scope: Normalised 90-day chart (all 3 indexed to 100), key ratios table,
signals summary table (metal | price | signal | confidence | regime).

**Step 6E — Portfolio Tracker: NOT STARTED**
Scope: session-state P&L for 3 metals, USD + AED, total row, disclaimer.

---

## Remaining Work Order (do not skip or reorder)

1. **Step 6D** — Multi-Metal Comparison Panel
4. **Step 6E** — Portfolio Tracker
5. **Step 7** — SHAP Transparency Layer
6. **Step 8** — KPI Command Centre
7. **Step 9** — GitHub repo rename (manual)
8. **Step 10** — Final commit + local test checklist
9. **Step 11** — Streamlit Cloud deployment (manual)

---

### Pending Manual Actions (do not automate)
- **Step 9**: GitHub repo rename → `apex-metals-ai`
  URL: github.com/hadihbk-png/gold-ai-trading-intelligence → Settings → General → Repository name
- **Step 11**: Streamlit Cloud subdomain → `apex-metals-ai`
  URL: share.streamlit.io → app Settings → General → subdomain

---

## Remaining Steps (verbatim from build prompt)

### STEP 5 — DECISION INTELLIGENCE CENTRE

Add new section in app.py:
"🎯 Decision Intelligence Centre"
Place immediately after the signal/confidence display.
Only show for Gold (selected_metal == Gold) for now —
will extend to Silver/Platinum in Step 6.

#### 5A: DAILY DECISION VERDICT

```python
def compute_decision_verdict(signal, confidence,
                              regime, rsi, bb_pctb,
                              rolling_accuracy,
                              no_trade_score,
                              classifier_consensus):
    reasons_for = []
    reasons_against = []

    if confidence >= 0.60:
        reasons_for.append(
            f"Strong model confidence ({confidence:.0%})")
    elif confidence < 0.50:
        reasons_against.append(
            f"Low confidence ({confidence:.0%})")

    if rolling_accuracy >= 0.40:
        reasons_for.append(
            f"Model in strong recent form "
            f"({rolling_accuracy:.0%} rolling 30d)")
    elif rolling_accuracy < 0.33:
        reasons_against.append(
            f"Model below random baseline recently "
            f"({rolling_accuracy:.0%})")

    if regime == 'high_vol':
        reasons_against.append(
            "High volatility regime — elevated uncertainty")
    elif regime == 'trending':
        reasons_for.append(
            "Trending regime — model performs best here")

    if signal == 'UP' and rsi > 75:
        reasons_against.append(
            "RSI overbought (>75) — mean reversion risk")
    if signal == 'DOWN' and rsi < 25:
        reasons_against.append(
            "RSI oversold (<25) — bounce risk")
    if bb_pctb > 0.9:
        reasons_against.append(
            "Price at upper Bollinger Band")
    if bb_pctb < 0.1:
        reasons_against.append(
            "Price at lower Bollinger Band")

    if classifier_consensus < 0.67:
        reasons_against.append(
            f"Classifiers split — low consensus "
            f"({classifier_consensus:.0%})")

    if no_trade_score >= 3:
        reasons_against.append(
            f"No-trade filter: {no_trade_score}/5 "
            f"conditions active")

    against = len(reasons_against)

    if against >= 3:
        return ("NO-TRADE","🔴","#ef4444",
                reasons_for, reasons_against)
    elif against >= 2:
        return ("CAUTION","🟡","#f59e0b",
                reasons_for, reasons_against)
    else:
        return ("GO","🟢","#22c55e",
                reasons_for, reasons_against)
```

Display as a large styled verdict card with:
- Verdict badge (GO/CAUTION/NO-TRADE) prominently
- Two columns: supporting reasons / against reasons
- Disclaimer: research framework only, not advice

#### 5B: ATR-BASED RESEARCH ZONES

```python
def compute_trade_zones(price, signal, atr):
    if signal == 'UP':
        entry_low  = price - atr * 0.3
        entry_high = price + atr * 0.1
        target     = price + atr * 2.0
        stop       = price - atr * 1.0
    elif signal == 'DOWN':
        entry_low  = price - atr * 0.1
        entry_high = price + atr * 0.3
        target     = price - atr * 2.0
        stop       = price + atr * 1.0
    else:
        return None

    reward = abs(target - price)
    risk   = abs(stop - price)
    rr     = round(reward/risk, 1) if risk > 0 else 0

    return {
        'entry': f"${entry_low:,.0f}–${entry_high:,.0f}",
        'target': f"${target:,.0f}",
        'stop': f"${stop:,.0f}",
        'rr': f"{rr}:1",
        'atr': f"${atr:,.0f}"
    }
```

Show as 4-column metric row with disclaimer below.

#### 5C: CLASSIFIER CONSENSUS

Extract individual model votes from the ensemble.
Display as 3 indicator pills:
XGBoost: [UP/SIDEWAYS/DOWN] ·
LightGBM: [UP/SIDEWAYS/DOWN] ·
CatBoost: [UP/SIDEWAYS/DOWN]

Consensus: [3/3 unanimous ✅] or [2/3 majority ⚠️]

---

### STEP 6 — SILVER & PLATINUM EXPANSION

This is the metals expansion. Build for both
Silver (XAG) and Platinum (XPT) simultaneously.

#### 6A: PRICE DATA — Silver & Platinum

Add to the data fetching layer (same waterfall pattern):

Silver price waterfall:
1. Try yfinance "SI=F" (silver futures)
2. Try yfinance "SLV" (silver ETF proxy × 10)
3. Try Alpha Vantage with XAG symbol
4. Fall back to yfinance "SI=F" close

Platinum price waterfall:
1. Try yfinance "PL=F" (platinum futures)
2. Try yfinance "PPLT" (platinum ETF proxy)
3. Fall back to yfinance "PL=F" close

For each metal, display:
- Spot price in selected currency (AED default)
- Daily change $ and %
- Data source label
- 90-day chart (same OHLC format as gold)

Gold/Silver ratio:
  gsr = gold_price / silver_price
  Display with historical context:
  "Historical avg: ~65 · Current: {gsr:.1f}"
  Flag: Above 80 = silver historically cheap vs gold
  Flag: Below 50 = gold historically cheap vs silver

Platinum/Gold spread:
  ptg_spread = gold_price - platinum_price
  Display: "Gold premium over platinum: ${spread:,.0f}"
  Historical context: normally platinum > gold

#### 6B: SIGNAL MODELS — Silver & Platinum

Train separate signal models for Silver and Platinum
using the same feature engineering pipeline as Gold
but with their respective OHLCV data.

Create src/train_silver.py and src/train_platinum.py
(or add functions to src/train.py):

```python
def train_metal_model(ticker, metal_name):
    """
    Train signal model for a given metal ticker.
    Returns trained model and training stats.
    """
    print(f"Training {metal_name} model...")

    # Fetch OHLCV data
    df = yf.download(ticker, period="5y",
                     progress=False)
    if df.empty or len(df) < 200:
        print(f"Insufficient data for {metal_name}")
        return None, None

    # Apply same feature engineering
    df = build_features(df)  # existing function
    df = classify_regime(df)

    # Train ensemble (same architecture as gold)
    model, stats = train_ensemble(df)

    return model, stats
```

Save in models.pkl:
```python
{
  'gold': {
    'ensemble': ..., 'regime_models': ...,
    'thresholds': ..., 'feature_names': ...,
    'trained_at': ..., 'training_stats': ...
  },
  'silver': {
    'ensemble': ..., 'regime_models': ...,
    'thresholds': {'down': 0.25, 'up': 0.38},
    'feature_names': ..., 'trained_at': ...,
    'training_stats': ...
  },
  'platinum': {
    'ensemble': ..., 'regime_models': ...,
    'thresholds': {'down': 0.25, 'up': 0.38},
    'feature_names': ..., 'trained_at': ...,
    'training_stats': ...
  }
}
```

Note: Silver and Platinum may have less liquid data
than gold. If training data < 200 samples for any
regime model, skip that regime model and use
universal ensemble only. Never crash — graceful
degradation always.

STOP AFTER TRAINING. Report:
- Silver model: overall accuracy, per-class accuracy
- Platinum model: overall accuracy, per-class accuracy
- Any data issues (gaps, insufficient samples)

Wait for user confirmation before Step 7.

#### 6C: FULL DASHBOARD FOR SILVER & PLATINUM

When selected_metal == Silver or Platinum, show
the full dashboard with all sections:

For each metal show:
1. Price header (same layout as gold)
2. Signal card (UP/SIDEWAYS/DOWN)
3. Model confidence
4. Market regime
5. Morning Brief — update src/explainer.py to
   accept metal parameter:

```python
   def generate_morning_brief(metal, price, signal,
                               confidence, ...):
       if metal == 'silver':
           title = "XAG/USD Morning Brief"
           context = "silver spot price"
       elif metal == 'platinum':
           title = "XPT/USD Morning Brief"
           context = "platinum spot price"
       ...
```

6. 90-day price chart
7. Decision Intelligence Centre
   (same verdict framework, metal-specific zones)
8. Feature importance chart
9. Signal explanation

#### 6D: MULTI-METAL COMPARISON PANEL

Add a new section visible when ANY metal is selected:
"⚖️ Multi-Metal Comparison"

Sub-panel 1: Normalised price chart (90 days)
- All 3 metals indexed to 100 at start of period
- Plotly line chart: Gold (amber), Silver (grey),
  Platinum (purple)
- Shows relative performance at a glance

Sub-panel 2: Key ratios (3 metrics)
- Gold/Silver Ratio: current vs 1yr avg
- Platinum/Gold Spread: current vs 1yr avg
- Silver/Platinum Ratio: current vs 1yr avg

Sub-panel 3: Signals summary table
```
Metal | Price (USD) | Signal | Confidence | Regime
Gold  | $X,XXX      | UP     | 50.2%      | Neutral
Silver| $XX.XX      | ...    | ...        | ...
Plat. | $X,XXX      | ...    | ...        | ...
```

#### 6E: PORTFOLIO TRACKER

Add expander "💼 My Portfolio Tracker" in app.py.
Use st.session_state for portfolio data storage
(persists during session — clearly note it resets
on page refresh).

Input form (3 metal sections):
For each metal (Gold, Silver, Platinum):
```python
  col1, col2, col3 = st.columns(3)
  with col1: qty = st.number_input(
      f"{metal} quantity (oz)", min_value=0.0,
      step=0.1, key=f"qty_{metal}")
  with col2: entry = st.number_input(
      f"Entry price (USD/oz)", min_value=0.0,
      step=1.0, key=f"entry_{metal}")
  with col3: entry_date = st.date_input(
      f"Entry date", key=f"date_{metal}")
```

Compute live P&L:
```python
  current_price = get_metal_price(metal)
  position_value_usd = qty * current_price
  cost_basis_usd = qty * entry_price
  pnl_usd = position_value_usd - cost_basis_usd
  pnl_pct = (pnl_usd / cost_basis_usd * 100
             if cost_basis_usd > 0 else 0)

  # Convert to AED
  position_value_aed = position_value_usd * 3.6725
  pnl_aed = pnl_usd * 3.6725
```

Display summary table:
```
Metal | Qty (oz) | Entry | Current | Value USD |
Value AED | P&L USD | P&L AED | P&L %
```

Total portfolio row at bottom:
Total value USD · Total value AED · Total P&L USD ·
Total P&L AED · Total P&L %

Color code: Green P&L = profit, Red = loss

Large disclaimer:
"Portfolio tracker uses live market prices for
research purposes only. Values are indicative.
This does not constitute financial advice."

---

### STEP 7 — SHAP TRANSPARENCY LAYER

Add shap to requirements.txt if not present.

For the currently selected metal's active signal,
add expander "🔬 Why did the model decide this?"

```python
def compute_shap_explanation(model, X_live,
                              feature_names,
                              pred_class,
                              top_n=12):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_live)
        if isinstance(shap_values, list):
            sv = shap_values[pred_class][0]
        else:
            sv = shap_values[0]
        df_shap = pd.DataFrame({
            'feature': feature_names,
            'contribution': sv,
            'abs': abs(sv)
        }).sort_values('abs', ascending=False).head(top_n)
        return df_shap
    except Exception:
        return None
```

Display as Plotly horizontal bar chart:
- Green bars: features pushing TOWARD the signal
- Red bars: features pushing AGAINST the signal
- Y axis: feature names (clean, readable)
- X axis: SHAP contribution value
- Title: "What drove today's [SIGNAL] signal —
  top 12 features"
- Caption: "SHAP values show each feature's
  contribution to this specific prediction."

---

### STEP 8 — KPI COMMAND CENTRE

Add expander "📊 KPI Command Centre" in app.py.
This reads from data/live_validation_log.csv.

Update audit trail logging to capture per-signal:
```
timestamp, metal, signal, confidence, price_at_signal,
regime, verdict, classifier_consensus,
top_feature_1, top_feature_2, top_feature_3,
price_24h_later, correct, pnl_hypothetical
```

Add update_audit_trail() function that:
1. Reads existing log
2. For yesterday's entry: fills price_24h_later,
   correct, pnl_hypothetical using today's price
3. Appends today's new entry
4. Saves back to CSV

KPI display (3 rows of 4 metrics each):

Row 1 — Accuracy:
Rolling 30d accuracy · Overall accuracy ·
High-confidence accuracy (>60%) · GO verdict accuracy

Row 2 — Signal breakdown:
DOWN accuracy · SIDEWAYS accuracy ·
UP accuracy · Total signals logged

Row 3 — Health:
Consecutive correct signals · Model age (days) ·
Days since last retrain · Data source uptime %

Hypothetical equity curve (clearly labelled):
"HYPOTHETICAL RESEARCH SIMULATION — NOT REAL
PERFORMANCE — NOT FINANCIAL ADVICE"
Plotly line chart: cumulative result if all GO
signals were followed, +1 correct / -1 incorrect.
Starting value: 100 (index).
Label each drawdown period clearly.

---

### STEP 9 — GITHUB REPOSITORY RENAME

After all code is complete and tested locally:

1. Update git remote URL:
   ```
   git remote set-url origin \
   https://github.com/hadihbk-png/apex-metals-ai.git
   ```

2. Verify: `git remote -v`

**MANUAL ACTION REQUIRED — DO NOT AUTOMATE:**
Before pushing, manually rename the GitHub repository:
- Go to: github.com/hadihbk-png/gold-ai-trading-intelligence
- Settings → General → Repository name
- Change to: apex-metals-ai
- Click "Rename"
Then confirm to continue with push.

---

### STEP 10 — FINAL COMMIT AND PUSH

Test locally first:
```
streamlit run app.py
```

Confirm ALL of the following work:
- [ ] Landing page displays correctly
- [ ] "Launch Dashboard" button enters the dashboard
- [ ] Metal selector tabs (Gold/Silver/Platinum)
- [ ] Gold dashboard fully functional
- [ ] Silver dashboard shows price + signal
- [ ] Platinum dashboard shows price + signal
- [ ] Multi-metal comparison panel renders
- [ ] Portfolio tracker input and P&L calculation
- [ ] Decision Intelligence Centre verdict
- [ ] ATR research zones display
- [ ] SHAP waterfall chart renders
- [ ] KPI Command Centre metrics load
- [ ] No error banners or crashes
- [ ] "← Back to overview" returns to landing page

Then commit:
```
git add -A
git commit -m "APEX Metals AI v6.0 — Complete platform
rebuild: rename, landing page, regime-conditional models,
macro features, Silver & Platinum expansion, Decision
Intelligence Centre, SHAP transparency, Portfolio tracker,
KPI Command Centre"
git push origin main
```

---

### STEP 11 — STREAMLIT CLOUD DEPLOYMENT

After push confirmed, instruct user to:

1. Go to share.streamlit.io
2. Find current app → Settings → General
3. Change subdomain to: apex-metals-ai
4. New URL: apex-metals-ai.streamlit.app
5. Reboot the app
6. Confirm deployment successful

Also update Streamlit Cloud app settings:
- App name: "APEX Metals AI"
- Main file: app.py (unchanged)
- All secrets remain the same (no changes needed)

**MANDATORY STOPS — DO NOT SKIP:**
- STOP 2: After Step 6B — report Silver & Platinum accuracy
- STOP 3: After Step 10 local test — confirm checklist before final commit
