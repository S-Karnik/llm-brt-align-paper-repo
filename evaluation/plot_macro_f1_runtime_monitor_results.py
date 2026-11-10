#!/usr/bin/env python3
"""Plot runtime-monitor macro-F1 under 50/50 class reweighting."""

import argparse
import glob
import json
import os
from collections import defaultdict
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import directories

DEFAULT_RESULTS_ROOT = os.path.join(directories.DATA_CACHE_DIR, "final_runtime_monitor_results")
MODEL_NAMES = {
    "Qwen/Qwen2-1.5B": "Qwen2-1.5B",
    "meta-llama/Llama-2-7b-hf": "Llama2-7b",
    "mistralai/Ministral-8B-Instruct-2410": "Ministral-8B",
    "tiiuae/falcon-7b-instruct": "Falcon-7B",
    "openai/gpt-oss-20b": "gpt-oss-20b",
}
METHODS = [
    "SaP",
    "Re-Control",
    "BRT",
    "Sample-BRT",
    "Prefix Classifier",
    "Generate-Then-Classify",
]
METHOD_COLORS = {
    "SaP": "#806363",
    "Re-Control": "#477998",
    "BRT": "#d46322",
    "Sample-BRT": "#ff9500",
    "Prefix Classifier": "#2a9d8f",
    "Generate-Then-Classify": "#6a4c93",
}
DATASETS = ["beavertails", "real_toxicity", "ultrasafety"]


def macro_f1_from_class_recalls(safe_success, unsafe_success):
    """Compute macro-F1 after assigning equal mass to safe and unsafe classes."""
    unsafe_true_positive = unsafe_success
    unsafe_false_negative = 1.0 - unsafe_success
    unsafe_false_positive = 1.0 - safe_success
    unsafe_denominator = (
        2.0 * unsafe_true_positive
        + unsafe_false_positive
        + unsafe_false_negative
    )
    f1_unsafe = (
        2.0 * unsafe_true_positive / unsafe_denominator
        if unsafe_denominator > 0
        else 0.0
    )

    safe_true_positive = safe_success
    safe_false_negative = 1.0 - safe_success
    safe_false_positive = 1.0 - unsafe_success
    safe_denominator = (
        2.0 * safe_true_positive
        + safe_false_positive
        + safe_false_negative
    )
    f1_safe = (
        2.0 * safe_true_positive / safe_denominator
        if safe_denominator > 0
        else 0.0
    )
    return 0.5 * (f1_safe + f1_unsafe)


def load_seed_scores(results_root, model_name, aggregate_datasets, dataset_filter):
    filename = model_name.replace("/", "_") + ".json"
    paths = sorted(glob.glob(os.path.join(results_root, "seed_*", filename)))
    if not paths:
        raise FileNotFoundError(
            f"No runtime-monitor results named {filename} under {results_root}"
        )

    scores = defaultdict(dict)
    for path in paths:
        seed = os.path.basename(os.path.dirname(path))
        with open(path) as input_file:
            results = json.load(input_file)

        for method in METHODS:
            method_results = results.get(method, {})
            if aggregate_datasets:
                dataset_scores = []
                for dataset in DATASETS:
                    metrics = method_results.get(dataset)
                    if metrics is None:
                        continue
                    dataset_scores.append(
                        macro_f1_from_class_recalls(
                            metrics["total_safe_classification_rate"],
                            metrics["total_unsafe_classification_rate"],
                        )
                    )
                if dataset_scores:
                    scores[method][seed] = mean(dataset_scores)
            else:
                metrics = method_results.get(dataset_filter)
                if metrics is not None:
                    scores[method][seed] = macro_f1_from_class_recalls(
                        metrics["total_safe_classification_rate"],
                        metrics["total_unsafe_classification_rate"],
                    )
    return scores


def mean_std(values):
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default=DEFAULT_RESULTS_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--aggregate_datasets",
        action="store_true",
        help="Macro-average the three dataset scores within each seed.",
    )
    mode.add_argument(
        "--dataset_filter",
        choices=DATASETS,
        help="Plot one dataset instead of averaging datasets.",
    )
    parser.add_argument("--save_path", default="macro_f1_scores.pdf")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    summaries = defaultdict(dict)
    for raw_model, display_model in MODEL_NAMES.items():
        try:
            seed_scores = load_seed_scores(
                args.results_root,
                raw_model,
                args.aggregate_datasets,
                args.dataset_filter,
            )
        except FileNotFoundError as error:
            print(f"Skipping {raw_model}: {error}")
            continue

        for method, values_by_seed in seed_scores.items():
            values = list(values_by_seed.values())
            if not values:
                continue
            score_mean, score_std = mean_std(values)
            summaries[display_model][method] = (
                score_mean,
                score_std,
                len(values),
            )
            print(
                f"{display_model} {method}: {score_mean:.6f} +/- "
                f"{score_std:.6f} ({len(values)} seeds)"
            )

    models = [
        display_name
        for display_name in MODEL_NAMES.values()
        if display_name in summaries
    ]
    if not models:
        raise RuntimeError("No runtime-monitor results were available to plot.")

    y = np.arange(len(models), dtype=float)
    bar_height = 0.12
    offsets = np.linspace(-1.5, 1.5, len(METHODS)) * bar_height

    figure, axis = plt.subplots(figsize=(12, 6.5))
    for offset, method in zip(offsets, METHODS):
        means = []
        errors = []
        for model in models:
            score_mean, score_std, _ = summaries[model].get(
                method, (np.nan, 0.0, 0)
            )
            means.append(score_mean)
            errors.append(score_std)

        means = np.asarray(means)
        errors = np.asarray(errors)
        valid = np.isfinite(means)
        axis.barh(
            y[valid] + offset,
            means[valid],
            height=bar_height,
            xerr=errors[valid],
            capsize=3,
            color=METHOD_COLORS[method],
            label=method,
            alpha=0.92,
        )

    axis.set_yticks(y)
    axis.set_yticklabels([])
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.tick_params(axis="x", labelbottom=False)
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.grid(False)
    figure.tight_layout()
    figure.savefig(args.save_path, dpi=300, bbox_inches="tight")
    print(f"Saved macro-F1 plot to {args.save_path}")


if __name__ == "__main__":
    main()
