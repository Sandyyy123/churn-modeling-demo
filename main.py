"""Run the full churn sample: python main.py [--customers N] [--seed S] [--out DIR]"""
import argparse
import json

from churnkit.pipeline import run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--customers", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    res = run(a.customers, a.seed, a.out)
    s, m = res["summary"], res["metrics"]
    print(json.dumps({"rows": s["rows"], "churn_rate": s["churn_rate"], "censoring_demo": s["censoring_demo"],
                      "final_model": m["final_model"], "leakage_demo": m["leakage_demo"],
                      "test_metrics": {k: {x: round(v[x], 4) for x in ("roc_auc", "pr_auc", "brier", "brier_skill", "ece",
                                                                         "top_decile_lift")}
                                       for k, v in m["models"].items()}}, indent=2))
    print(f"Wrote outputs to {a.out}/")


if __name__ == "__main__":
    main()
