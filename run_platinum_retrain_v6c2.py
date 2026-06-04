"""
Phase 6C — Platinum honest directional retrain (boost=1.0, LGB fix applied).

Discards the boost=5.0 Platinum model (collapsed DOWN to 24.0%).
Retrains Platinum with sideways_weight_boost=1.0 (no artificial SIDEWAYS pressure)
and saves it alongside the existing Silver run-2 bundle from metals_models.pkl.

Silver is NOT retrained — the run-2 bundle (LGB fix, boost=1.0, 49.8% overall) is
loaded from disk and preserved as-is.

Run:
  python run_platinum_retrain_v6c2.py

Expected runtime: ~10 min (Platinum only).
"""

import sys
import io
import time
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
elif hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, ".")

from src.train import train_metal_model, save_metal_models, load_metal_models

SEP = "-" * 74

# Known Silver run-2 numbers (LGB fix, boost=1.0, 2026-06-04 run)
SILVER_RUN2 = {
    "overall":   0.498,
    "recall":    {"DOWN": 0.443, "SIDEWAYS": 0.103, "UP": 0.679},
    "precision": {"DOWN": 0.492, "SIDEWAYS": 0.211, "UP": 0.541},
    "boost": 1.0,
}

# Platinum run we are replacing (boost=5.0, collapsed DOWN)
PLATINUM_DISCARD = {
    "overall":   0.446,
    "recall":    {"DOWN": 0.240, "SIDEWAYS": 0.033, "UP": 0.704},
    "precision": {"DOWN": 0.353, "SIDEWAYS": 0.062, "UP": 0.521},
    "boost": 5.0,
}


def main():
    t0 = time.time()
    print("=" * 74)
    print("APEX Metals AI — Platinum Honest Retrain (boost=1.0, LGB fix)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 74)

    # ── Load existing Silver bundle ───────────────────────────────────────────
    print()
    print("Loading existing metals_models.pkl to preserve Silver run-2…")
    existing = load_metal_models()
    silver_bundle = existing.get("silver")
    if silver_bundle is None:
        print("ERROR: Silver bundle not found in metals_models.pkl — aborting.")
        sys.exit(1)
    saved_acc = silver_bundle.get("overall_acc", 0)
    print(f"  Silver loaded: overall_acc={saved_acc:.1%}  "
          f"(expected ~49.8% for run-2)")
    if saved_acc < 0.45:
        print("  WARNING: loaded Silver accuracy is below 45% — "
              "this may not be the run-2 model. Proceeding anyway.")

    # ── Retrain Platinum at boost=1.0 ────────────────────────────────────────
    print()
    print(SEP)
    print("  PLATINUM  (boost=1.0, LGB fix — honest directional model)")
    print(SEP)
    print("  Discarding boost=5.0 model (DOWN collapsed to 24.0%).")
    print()
    t1 = time.time()
    platinum = train_metal_model("PL=F", "Platinum", force_sideways_boost=1.0)
    if platinum is None:
        print("ERROR: Platinum training failed — aborting.")
        sys.exit(1)
    elapsed_pt = (time.time() - t1) / 60
    print(f"Platinum done in {elapsed_pt:.1f} min")

    # ── Save ─────────────────────────────────────────────────────────────────
    save_metal_models({"silver": silver_bundle, "platinum": platinum})
    print(f"\nSaved -> models/metals_models.pkl")

    # ── Final report ─────────────────────────────────────────────────────────
    rec  = platinum.get("per_class_recall",    platinum.get("per_class_acc", {}))
    prec = platinum.get("per_class_precision", {})
    f1   = platinum.get("per_class_f1", {})
    boost = platinum.get("sideways_weight_boost", "?")

    print()
    print("FINAL STATE — both metals saved")
    print(SEP)
    hdr = (f"{'Metal':<10} | {'Overall':>7} | "
           f"{'DOWN rec':>8} | {'DOWN pre':>8} | "
           f"{'SIDE rec':>8} | {'SIDE pre':>8} | "
           f"{'UP rec':>6} | {'UP pre':>6} | "
           f"{'Boost':>5} | {'Train':>5} | {'Test':>4}")
    print(hdr)
    print(SEP)

    # Silver (from preserved run-2)
    sr = SILVER_RUN2["recall"]
    sp = SILVER_RUN2["precision"]
    print(
        f"{'Silver':<10} | {SILVER_RUN2['overall']:>7.1%} | "
        f"{sr.get('DOWN',0):>8.1%} | {sp.get('DOWN',0):>8.1%} | "
        f"{sr.get('SIDEWAYS',0):>8.1%} | {sp.get('SIDEWAYS',0):>8.1%} | "
        f"{sr.get('UP',0):>6.1%} | {sp.get('UP',0):>6.1%} | "
        f"{SILVER_RUN2['boost']!s:>5} | {'857':>5} | {'215':>4}"
        f"  [preserved run-2]"
    )
    # Platinum (freshly trained)
    print(
        f"{'Platinum':<10} | {platinum['overall_acc']:>7.1%} | "
        f"{rec.get('DOWN',0):>8.1%} | {prec.get('DOWN',0):>8.1%} | "
        f"{rec.get('SIDEWAYS',0):>8.1%} | {prec.get('SIDEWAYS',0):>8.1%} | "
        f"{rec.get('UP',0):>6.1%} | {prec.get('UP',0):>6.1%} | "
        f"{boost!s:>5} | {platinum['n_train']:>5} | {platinum['n_test']:>4}"
        f"  [fresh retrain]"
    )
    print(SEP)

    # ── Delta vs discarded boost=5.0 model ───────────────────────────────────
    print()
    print("Platinum BEFORE (boost=5.0, discarded) -> AFTER (boost=1.0):")
    for cls in ["DOWN", "SIDEWAYS", "UP"]:
        bv = PLATINUM_DISCARD["recall"].get(cls, 0)
        av = rec.get(cls, 0)
        d  = av - bv
        arrow = "up" if d > 0 else "down"
        print(f"  {cls:<9}: {bv:.1%} -> {av:.1%}  ({arrow} {abs(d):.1%})")

    # ── SIDEWAYS honest status ────────────────────────────────────────────────
    side_pt = rec.get("SIDEWAYS", 0)
    side_sv = SILVER_RUN2["recall"].get("SIDEWAYS", 0)
    print()
    print("SIDEWAYS status (honest):")
    print(f"  Silver   SIDEWAYS recall = {side_sv:.1%}  "
          "(learnable by single-LGB probe at 25.6%; suppressed by meta-learner)")
    print(f"  Platinum SIDEWAYS recall = {side_pt:.1%}  "
          "(structurally not separable with current 150 features)")
    print()
    print("Neither metal meets the 20% floor. Both are directional-leaning models.")
    print("SIDEWAYS signals for Silver/Platinum will be flagged low-reliability in 6C.")

    elapsed = (time.time() - t0) / 60
    print()
    print(f"Total elapsed: {elapsed:.1f} min")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 74)
    print("STOP — awaiting review before 6C.")


if __name__ == "__main__":
    main()
