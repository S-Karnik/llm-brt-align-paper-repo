#!/usr/bin/env python3
import argparse
import glob
import json
import os
from statistics import mean, stdev

from tabulate import tabulate

import directories

DEFAULT_RESULTS_ROOT = os.path.join(directories.DATA_CACHE_DIR, "final_runtime_monitor_auc_results")


def mean_std(values):
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()

    model_file = args.model_name.replace("/", "_") + ".json"
    paths = sorted(
        glob.glob(os.path.join(args.results_root, "seed_*", model_file))
    )
    if not paths:
        raise FileNotFoundError(
            f"No AUC result files named {model_file} under {args.results_root}"
        )

    grouped = {}
    for path in paths:
        seed = os.path.basename(os.path.dirname(path))
        with open(path) as input_file:
            results = json.load(input_file)
        for method, datasets in results.items():
            for dataset, metrics in datasets.items():
                grouped.setdefault((method, dataset), {})[seed] = metrics

    rows = []
    for (method, dataset), seed_results in sorted(grouped.items()):
        auroc = [result["auroc"] for result in seed_results.values()]
        auprc = [result["auprc"] for result in seed_results.values()]
        average_precision = [
            result["average_precision"] for result in seed_results.values()
        ]
        prevalence = [
            result["unsafe_prevalence"] for result in seed_results.values()
        ]
        auroc_mean, auroc_std = mean_std(auroc)
        auprc_mean, auprc_std = mean_std(auprc)
        ap_mean, ap_std = mean_std(average_precision)
        prevalence_mean, _ = mean_std(prevalence)
        rows.append(
            [
                method,
                dataset,
                len(seed_results),
                f"{auroc_mean:.6f} +/- {auroc_std:.6f}",
                f"{auprc_mean:.6f} +/- {auprc_std:.6f}",
                f"{ap_mean:.6f} +/- {ap_std:.6f}",
                f"{prevalence_mean:.6f}",
            ]
        )

    print(
        tabulate(
            rows,
            headers=[
                "Method",
                "Dataset",
                "Seeds",
                "AUROC",
                "AUPRC",
                "Avg. Precision",
                "Unsafe Prev.",
            ],
            tablefmt="grid",
        )
    )


if __name__ == "__main__":
    main()
