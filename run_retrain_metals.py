"""
Train Silver and Platinum signal models and save to models/metals_models.pkl.
Usage:  python run_retrain_metals.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import train_metal_model, save_metal_models, load_metal_models


def _bar(title):
    print("\n" + "=" * 55)
    print("  " + title)
    print("=" * 55)


def main():
    results = load_metal_models()
    errors  = []

    for ticker, name, key in [
        ("SI=F", "Silver",   "silver"),
        ("PL=F", "Platinum", "platinum"),
    ]:
        _bar("TRAINING " + name.upper() + " (" + ticker + ")")
        bundle = train_metal_model(ticker, name)
        if bundle:
            results[key] = bundle
            pc = bundle["per_class_acc"]
            print("\nOK  " + name + " complete")
            print("   Overall accuracy : " + format(bundle["overall_acc"], ".1%"))
            print("   DOWN             : " + format(pc.get("DOWN", 0), ".1%"))
            print("   SIDEWAYS         : " + format(pc.get("SIDEWAYS", 0), ".1%"))
            print("   UP               : " + format(pc.get("UP", 0), ".1%"))
            print("   Train bars       : " + str(bundle["n_train"]))
            print("   Test bars        : " + str(bundle["n_test"]))
        else:
            errors.append(name)
            print("\nFAIL  " + name + " training failed -- check output above")

    _bar("SAVING")
    if results:
        save_metal_models(results)
        print("Saved: " + str(sorted(results.keys())) + " -> models/metals_models.pkl")
    else:
        print("Nothing to save.")

    if errors:
        print("\n!! Failed: " + str(errors))
        sys.exit(1)

    print("\nDONE  All metal models trained successfully.")


if __name__ == "__main__":
    main()
