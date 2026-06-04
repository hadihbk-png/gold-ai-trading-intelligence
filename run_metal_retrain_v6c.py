"""
Phase 6C pre-requisite: Metal SIDEWAYS rebalance retrain.

Fixes applied in this run
  1. classify_regime() now receives the full OHLCV df — regime routing works
     for Silver and Platinum (previously every row fell back to 'neutral').
  2. Per-metal sideways_weight_boost grid search [1.0 … 5.0] — picks the
     smallest value that lifts SIDEWAYS recall to ≥ 20 % without collapsing
     DOWN or UP recall below 15 %.
  3. Full retrain with the chosen boost.  Reports precision + recall + F1
     on a strictly-forward 80/20 time-series split (no look-ahead).

Run:
  python run_metal_retrain_v6c.py

Expected runtime: 25 – 40 min
  ~2 min  data fetch + feature engineering per metal
  ~1 min  boost probe grid (9 × single-LGB) per metal
  ~10 min full ensemble retrain per metal (fast mode: fixed hyperparams,
          3 OOF folds, LSTM)
"""

import sys
import io
import time
from datetime import datetime

# Force UTF-8 output so Unicode table characters survive Windows cp1252 pipes
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
elif hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, ".")

from src.train import train_metal_model, save_metal_models

# ── BEFORE state (Phase 6B training — recall only, precision unknown) ─────────
# Source: PHASE6_STATE.md, trained with sideways_weight_boost=1.0
BEFORE = {
    "Silver": {
        "overall":   0.372,
        "recall":    {"DOWN": 0.343, "SIDEWAYS": 0.128, "UP": 0.481},
        "precision": None,   # not recorded in prior run
        "n_train":   857,
        "n_test":    215,
        "boost":     1.0,
    },
    "Platinum": {
        "overall":   0.427,
        "recall":    {"DOWN": 0.413, "SIDEWAYS": 0.000, "UP": 0.556},
        "precision": None,
        "n_train":   851,
        "n_test":    213,
        "boost":     1.0,
    },
}

SEP = "-" * 74


def _print_before_table():
    print()
    print("BEFORE  (Phase 6B — sideways_weight_boost=1.0, recall only)")
    print(SEP)
    hdr = (f"{'Metal':<10} |{'Overall':>7} |"
           f"{'DOWN rec':>8} |{'DOWN pre':>8} |"
           f"{'SIDE rec':>8} |{'SIDE pre':>8} |"
           f"{'UP rec':>6} |{'UP pre':>6} |{'Train':>5} |{'Test':>4}")
    print(hdr)
    print(SEP)
    for name, b in BEFORE.items():
        r = b["recall"]
        p = b["precision"] or {}
        print(
            f"{name:<10} |{b['overall']:>7.1%} |"
            f"{r.get('DOWN',0):>8.1%} |{'N/A':>8} |"
            f"{r.get('SIDEWAYS',0):>8.1%} |{'N/A':>8} |"
            f"{r.get('UP',0):>6.1%} |{'N/A':>6} |"
            f"{b['n_train']:>5} |{b['n_test']:>4}"
        )
    print(SEP)


def _print_after_table(bundles: dict):
    print()
    print("AFTER  (this run — precision + recall on forward test split)")
    print(SEP)
    hdr = (f"{'Metal':<10} |{'Overall':>7} |"
           f"{'DOWN rec':>8} |{'DOWN pre':>8} |"
           f"{'SIDE rec':>8} |{'SIDE pre':>8} |"
           f"{'UP rec':>6} |{'UP pre':>6} |"
           f"{'Boost':>5} |{'Train':>5} |{'Test':>4}")
    print(hdr)
    print(SEP)
    for name, bnd in bundles.items():
        rec   = bnd.get("per_class_recall",    bnd.get("per_class_acc", {}))
        prec  = bnd.get("per_class_precision", {})
        boost = bnd.get("sideways_weight_boost", "?")
        print(
            f"{name:<10} |{bnd['overall_acc']:>7.1%} |"
            f"{rec.get('DOWN',0):>8.1%} |{prec.get('DOWN',0):>8.1%} |"
            f"{rec.get('SIDEWAYS',0):>8.1%} |{prec.get('SIDEWAYS',0):>8.1%} |"
            f"{rec.get('UP',0):>6.1%} |{prec.get('UP',0):>6.1%} |"
            f"{boost!s:>5} |{bnd['n_train']:>5} |{bnd['n_test']:>4}"
        )
    print(SEP)


def _print_delta_table(bundles: dict):
    print()
    print("BEFORE → AFTER  (recall delta; ↑ better, ↓ worse)")
    print(SEP)
    for name, bnd in bundles.items():
        b_rec = BEFORE[name]["recall"]
        a_rec = bnd.get("per_class_recall", bnd.get("per_class_acc", {}))
        boost = bnd.get("sideways_weight_boost", "?")
        b_ov  = BEFORE[name]["overall"]
        a_ov  = bnd["overall_acc"]
        print(f"  {name}  (boost {BEFORE[name]['boost']} → {boost}):")
        d = a_ov - b_ov
        print(f"    Overall   : {b_ov:.1%} → {a_ov:.1%}  "
              f"({'↑' if d>0 else '↓'} {abs(d):.1%})")
        for cls in ["DOWN", "SIDEWAYS", "UP"]:
            bv = b_rec.get(cls, 0)
            av = a_rec.get(cls, 0)
            d  = av - bv
            flag = " ✓" if cls == "SIDEWAYS" and av >= 0.20 else (
                   " ✗" if cls == "SIDEWAYS" and av < 0.20 else "")
            print(f"    {cls:<9}: {bv:.1%} → {av:.1%}  "
                  f"({'↑' if d>0 else '↓'} {abs(d):.1%}){flag}")
    print(SEP)


def main():
    t0 = time.time()
    print("=" * 74)
    print("APEX Metals AI — Phase 6C Pre-requisite: Metal SIDEWAYS Rebalance")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 74)

    _print_before_table()

    bundles = {}

    # ── Silver ────────────────────────────────────────────────────────────────
    print()
    print(f"{'─'*30}  SILVER  {'─'*30}")
    t1 = time.time()
    silver = train_metal_model("SI=F", "Silver")
    if silver is None:
        print("ERROR: Silver training failed — aborting")
        sys.exit(1)
    bundles["Silver"] = silver
    print(f"Silver done in {(time.time()-t1)/60:.1f} min")

    # ── Platinum ──────────────────────────────────────────────────────────────
    print()
    print(f"{'─'*30}  PLATINUM  {'─'*30}")
    t2 = time.time()
    platinum = train_metal_model("PL=F", "Platinum")
    if platinum is None:
        print("ERROR: Platinum training failed — aborting")
        sys.exit(1)
    bundles["Platinum"] = platinum
    print(f"Platinum done in {(time.time()-t2)/60:.1f} min")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_metal_models({"silver": silver, "platinum": platinum})
    print(f"\nSaved → models/metals_models.pkl")

    # ── Report ────────────────────────────────────────────────────────────────
    _print_after_table(bundles)
    _print_delta_table(bundles)

    # ── SIDEWAYS floor check ──────────────────────────────────────────────────
    print()
    print("SIDEWAYS recall floor check (≥ 20 % required before 6C):")
    all_pass = True
    for name, bnd in bundles.items():
        rec   = bnd.get("per_class_recall", bnd.get("per_class_acc", {}))
        side  = rec.get("SIDEWAYS", 0)
        ok    = side >= 0.20
        all_pass = all_pass and ok
        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"  {name}: SIDEWAYS recall = {side:.1%}  [{status}]")

    print()
    if all_pass:
        print("Both metals meet the SIDEWAYS floor. Awaiting your review before 6C.")
    else:
        print("One or more metals did NOT meet the SIDEWAYS floor.")
        print("Options: (a) accept and note limitation, (b) try higher boosts, "
              "(c) apply per-class threshold tuning on top of current model.")

    elapsed = (time.time() - t0) / 60
    print()
    print(f"Total elapsed: {elapsed:.1f} min")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 74)
    print("STOP — awaiting user review before Step 6C.")


if __name__ == "__main__":
    main()
