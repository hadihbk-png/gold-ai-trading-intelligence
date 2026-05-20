# Gold AI — Bug Fix Log

UI and reporting clarity fixes only. No model logic, retraining, model artifacts,
Dashboard, or Risk Management files were changed.

---

## Summary Table

| # | Date       | File                             | Category          | Issue                                      | Resolution                              |
|---|------------|----------------------------------|-------------------|--------------------------------------------|-----------------------------------------|
| 1 | 2026-05-20 | pages/3_Historical_Performance.py | Display cap       | Sortino Ratio could display absurdly large values (e.g. 847.532) | Capped display at >99.9 via `_fmt_ratio`; raw value unchanged |
| 2 | 2026-05-20 | pages/3_Historical_Performance.py | Misleading label  | Profit Factor `help` text missing; users could misread >999 | Added tooltip explaining cap and cause |
| 3 | 2026-05-20 | pages/3_Historical_Performance.py | Misleading label  | Live Backtest caption did not warn about in-sample bias | Caption now explicitly flags in-sample-only limitation |
| 4 | 2026-05-20 | pages/3_Historical_Performance.py | Deprecated API    | `width="stretch"` is deprecated in Streamlit | Replaced with `use_container_width=True` across all charts/tables |
| 5 | 2026-05-20 | pages/4_Live_Validation.py        | Missing context   | Validation table showed REAL and BACKFILLED rows with no visual distinction | Added `Source` column ("REAL" / "BACKFILLED") as first column |
| 6 | 2026-05-20 | pages/4_Live_Validation.py        | Confusing NaNs    | BACKFILLED rows showed raw NaN in signal/confidence/etc columns | NaN replaced with "—" in display copy only; CSV unchanged |
| 7 | 2026-05-20 | pages/4_Live_Validation.py        | Missing caveat    | No UI note confirming BACKFILLED rows are excluded from accuracy | Added caption below KPI metrics |
| 8 | 2026-05-20 | pages/4_Live_Validation.py        | Deprecated API    | `width="stretch"` deprecated | Replaced with `use_container_width=True` |

---

## Fix 1 — Sortino Ratio display cap

**File:** `pages/3_Historical_Performance.py`
**Issue:** When downside volatility is near zero (few or no down days), Sortino Ratio can
produce values like 847.532 which appear as legitimate measurements rather than
mathematical artefacts.
**Fix:** Added `_fmt_ratio(v, cap=99.9)` helper (display-only). Values above 99.9 render
as `>99.9`. The underlying `bt_m['Sortino Ratio']` value is never modified.

---

## Fix 2 & 3 — Metric help text and in-sample caption

**File:** `pages/3_Historical_Performance.py`
**Issue:** Profit Factor and Sortino KPI tiles had no help text. The Live Backtest section
caption did not warn readers that metrics were in-sample (the model was trained on the
same data).
**Fix:**
- `k6.metric("Sortino", ...)` now includes `help=` text about near-zero downside volatility.
- `k7.metric("Profit Factor", ...)` now includes `help=` text about the >999 display cap.
- Live Backtest caption extended with: "⚠️ In-sample only — model was trained on this
  data. Figures are optimistic relative to true out-of-sample performance."

---

## Fix 4 & 8 — Deprecated `width="stretch"` API

**Files:** `pages/3_Historical_Performance.py`, `pages/4_Live_Validation.py`
**Issue:** `st.plotly_chart(..., width="stretch")` and `st.dataframe(..., width="stretch")`
are deprecated Streamlit kwargs that generate warnings and may stop working in future
Streamlit versions.
**Fix:** Replaced all occurrences with `use_container_width=True`.

---

## Fix 5 — REAL vs BACKFILLED row distinction

**File:** `pages/4_Live_Validation.py`
**Issue:** The Validation Table showed REAL live predictions alongside BACKFILLED audit
gap rows with no visual distinction, making it impossible for users to tell which rows
had actual model signals.
**Fix:** A `Source` column is prepended to `display_df` at render time:
- `"REAL"` — row has a live model prediction
- `"BACKFILLED"` — row is an audit gap filled from market close prices only

`log_df` and `live_validation_log.csv` are not modified.

---

## Fix 6 — BACKFILLED NaN clarity

**File:** `pages/4_Live_Validation.py`
**Issue:** BACKFILLED rows showed `NaN` (or blank) in columns like `signal`, `confidence`,
`prob_up`, `regime`, `correct`, etc., which appeared as data errors rather than
intentional absences.
**Fix:** For BACKFILLED rows only, `NaN` values in the signal/probability/outcome columns
are replaced with `"—"` in `display_df`. The `log_df` dataframe and the CSV on disk
remain unchanged.

---

## Fix 7 — Accuracy exclusion caveat

**File:** `pages/4_Live_Validation.py`
**Issue:** The 5 accuracy KPI metrics gave no indication that BACKFILLED rows were
excluded. A reader could wonder whether the counts were inflated.
**Fix:** Added `st.caption(...)` below the KPI row:
> "Accuracy calculated on REAL predictions only. BACKFILLED audit rows have no signal
> and are excluded from all accuracy metrics."

This is consistent with the existing code behaviour: `_score_open_predictions` already
skips rows where `signal` is NaN (the BACKFILLED sentinel), and `scored` is filtered to
`log_df[log_df["correct"].notna()]` which further excludes them.

---

## Scope Confirmation

| Area                  | Changed? |
|-----------------------|----------|
| Model logic           | No       |
| Training / retraining | No       |
| Model artifacts       | No       |
| Dashboard page        | No       |
| Risk Management page  | No       |
| live_validation_log.csv schema | No |
| Accuracy calculation logic     | No |
| Backfill logic                 | No |
