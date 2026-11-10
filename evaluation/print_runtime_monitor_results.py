import argparse
import json
import os
import glob
from tabulate import tabulate
from statistics import mean, stdev

import directories

def mean_std(values):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        type=str,
        default=os.path.join(directories.DATA_CACHE_DIR, "final_runtime_monitor_results"),
                        help="Root dir containing seed_* subdirs")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name, e.g. openai/gpt-oss-20b (slashes replaced automatically)")
    parser.add_argument("--aggregate_datasets", action="store_true",
                        help="If set, averages across datasets for each method")
    args = parser.parse_args()

    parsed_model_name = args.model_name.replace("/", "_") + ".json"
    json_paths = glob.glob(os.path.join(args.results_root, "seed_*", parsed_model_name))
    if not json_paths:
        raise FileNotFoundError(f"No JSONs found for {parsed_model_name} under {args.results_root}")

    # metrics per (method, dataset, seed)
    per_seed = {}

    for jp in json_paths:
        print(f"Loading {jp}")
        seed = os.path.basename(os.path.dirname(jp))
        with open(jp, "r") as f:
            data = json.load(f)
        for method, datasets in data.items():
            if args.aggregate_datasets:
                # average across datasets within this seed
                agg_metrics = {m: [] for m in [
                    "total_classification_rate",
                    "total_safe_classification_rate",
                    "total_unsafe_classification_rate",
                    "unsafe_flag_rate_on_safe",
                    "f1_unsafe",
                ]}
                for metrics in datasets.values():
                    for m in agg_metrics:
                        if m == "unsafe_flag_rate_on_safe":
                            safe_rate = metrics.get("total_safe_classification_rate")
                            if safe_rate is not None:
                                agg_metrics[m].append(1.0 - safe_rate)
                        elif metrics.get(m) is not None:
                            agg_metrics[m].append(metrics[m])
                averaged = {m: mean(v) for m, v in agg_metrics.items() if v}
                per_seed.setdefault((method, "ALL"), {}).setdefault(seed, averaged)
            else:
                for dataset, metrics in datasets.items():
                    per_seed.setdefault((method, dataset), {}).setdefault(seed, {
                        "total_classification_rate": metrics.get("total_classification_rate", 0.0),
                        "total_safe_classification_rate": metrics.get("total_safe_classification_rate", 0.0),
                        "total_unsafe_classification_rate": metrics.get("total_unsafe_classification_rate", 0.0),
                        "unsafe_flag_rate_on_safe": 1.0 - metrics.get("total_safe_classification_rate", 0.0),
                        "f1_unsafe": metrics.get("f1_unsafe", 0.0),
                    })

    # regroup across seeds
    rows = []
    for (method, dataset), seeds_dict in sorted(per_seed.items()):
        row = [method, dataset]
        for m in ["total_classification_rate",
                  "total_safe_classification_rate",
                  "total_unsafe_classification_rate",
                  "unsafe_flag_rate_on_safe",
                  "f1_unsafe"]:
            vals = [metrics[m] for metrics in seeds_dict.values() if m in metrics]
            mu, sd = mean_std(vals)
            if "rate" in m:
                row.append(f"{100 * mu:.2f}% ± {100 * sd:.2f}%")
            else:
                row.append(f"{mu:.6f} ± {sd:.6f}")
        rows.append(row)

    headers = [
        "Method",
        "Dataset" if not args.aggregate_datasets else "Aggregated",
        "Cls. Success",
        "Safe Success",
        "Unsafe Success",
        "Safe Flagged Unsafe (FPR)",
        "F1 (Unsafe)",
    ]
    print(tabulate(rows, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main()
