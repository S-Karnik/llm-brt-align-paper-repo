#!/usr/bin/env python3
"""Print Macro-F1, AUROC, and AUPRC runtime-monitor results."""

import argparse
import glob
import json
import os
from pathlib import Path
from statistics import mean, stdev

from tabulate import tabulate

import directories

DEFAULT_FIXED_ROOT = os.path.join(directories.DATA_CACHE_DIR, "final_runtime_monitor_results")
DEFAULT_AUC_ROOT = os.path.join(directories.DATA_CACHE_DIR, "final_runtime_monitor_auc_results")
DEFAULT_MODELS = [
    "Qwen/Qwen2-1.5B",
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Ministral-8B-Instruct-2410",
    "tiiuae/falcon-7b-instruct",
    "openai/gpt-oss-20b",
]
MODEL_DISPLAY_NAMES = {
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
DATASETS = ["beavertails", "real_toxicity", "ultrasafety"]


def macro_f1_from_class_recalls(safe_success, unsafe_success):
    """Compute macro-F1 after assigning equal mass to both classes."""
    unsafe_true_positive = unsafe_success
    unsafe_false_negative = 1.0 - unsafe_success
    unsafe_false_positive = 1.0 - safe_success
    unsafe_denominator = (
        2.0 * unsafe_true_positive
        + unsafe_false_positive
        + unsafe_false_negative
    )
    unsafe_f1 = (
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
    safe_f1 = (
        2.0 * safe_true_positive / safe_denominator
        if safe_denominator > 0
        else 0.0
    )
    return 0.5 * (safe_f1 + unsafe_f1)


def load_seed_jsons(results_root, model_name):
    filename = model_name.replace("/", "_") + ".json"
    paths = sorted(glob.glob(os.path.join(results_root, "seed_*", filename)))
    results = {}
    for path in paths:
        seed_name = os.path.basename(os.path.dirname(path))
        if not seed_name.startswith("seed_"):
            continue
        seed = int(seed_name.removeprefix("seed_"))
        with open(path) as input_file:
            results[seed] = json.load(input_file)
    return results


def metric_triplet(fixed_metrics, auc_metrics):
    macro_f1 = macro_f1_from_class_recalls(
        fixed_metrics["total_safe_classification_rate"],
        fixed_metrics["total_unsafe_classification_rate"],
    )
    return {
        "macro_f1": macro_f1,
        "auroc": float(auc_metrics["auroc"]),
        "auprc": float(auc_metrics["auprc"]),
    }


def collect_model_results(
    fixed_root,
    auc_root,
    model_name,
    methods,
    datasets,
    aggregate_datasets,
):
    fixed_by_seed = load_seed_jsons(fixed_root, model_name)
    auc_by_seed = load_seed_jsons(auc_root, model_name)
    common_seeds = sorted(set(fixed_by_seed) & set(auc_by_seed))
    collected = {}

    for method in methods:
        if aggregate_datasets:
            per_seed = {}
            for seed in common_seeds:
                dataset_metrics = []
                for dataset in datasets:
                    fixed_metrics = fixed_by_seed[seed].get(method, {}).get(dataset)
                    auc_metrics = auc_by_seed[seed].get(method, {}).get(dataset)
                    if fixed_metrics is None or auc_metrics is None:
                        dataset_metrics = []
                        break
                    dataset_metrics.append(
                        metric_triplet(fixed_metrics, auc_metrics)
                    )
                if dataset_metrics:
                    per_seed[seed] = {
                        metric: mean(result[metric] for result in dataset_metrics)
                        for metric in ("macro_f1", "auroc", "auprc")
                    }
            if per_seed:
                collected[(method, "ALL")] = per_seed
        else:
            for dataset in datasets:
                per_seed = {}
                for seed in common_seeds:
                    fixed_metrics = fixed_by_seed[seed].get(method, {}).get(dataset)
                    auc_metrics = auc_by_seed[seed].get(method, {}).get(dataset)
                    if fixed_metrics is None or auc_metrics is None:
                        continue
                    per_seed[seed] = metric_triplet(fixed_metrics, auc_metrics)
                if per_seed:
                    collected[(method, dataset)] = per_seed

    return collected, common_seeds


def mean_std(values):
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def formatted_metric(per_seed, metric):
    values = [metrics[metric] for metrics in per_seed.values()]
    metric_mean, metric_std = mean_std(values)
    return f"{metric_mean:.3f} +/- {metric_std:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed_results_root", default=DEFAULT_FIXED_ROOT)
    parser.add_argument("--auc_results_root", default=DEFAULT_AUC_ROOT)
    parser.add_argument("--model_name", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--aggregate_datasets",
        action="store_true",
        help="Average the three datasets within each seed before summarizing seeds.",
    )
    mode.add_argument("--dataset_filter", choices=DATASETS)
    parser.add_argument(
        "--tablefmt",
        default="grid",
        help="Any tabulate format, such as grid, github, or latex_booktabs.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    datasets = [args.dataset_filter] if args.dataset_filter else DATASETS
    rows = []
    missing = []

    for model_name in args.model_name:
        collected, common_seeds = collect_model_results(
            args.fixed_results_root,
            args.auc_results_root,
            model_name,
            args.methods,
            datasets,
            args.aggregate_datasets,
        )
        if not common_seeds:
            missing.append(
                f"{model_name}: no seeds shared by fixed-threshold and AUC results"
            )
            continue

        dataset_order = ["ALL"] if args.aggregate_datasets else datasets
        for dataset in dataset_order:
            for method in args.methods:
                per_seed = collected.get((method, dataset))
                if not per_seed:
                    missing.append(
                        f"{model_name}/{dataset}/{method}: no complete paired results"
                    )
                    continue
                rows.append(
                    [
                        MODEL_DISPLAY_NAMES.get(model_name, model_name),
                        dataset,
                        method,
                        len(per_seed),
                        formatted_metric(per_seed, "macro_f1"),
                        formatted_metric(per_seed, "auroc"),
                        formatted_metric(per_seed, "auprc"),
                    ]
                )

    if not rows:
        raise RuntimeError("No paired runtime-monitor results were available.")

    table = tabulate(
        rows,
        headers=[
            "Model",
            "Dataset",
            "Method",
            "Seeds",
            "Macro-F1",
            "AUROC",
            "AUPRC",
        ],
        tablefmt=args.tablefmt,
    )
    print(table)
    if missing:
        print("\nMissing or incomplete combinations:")
        for message in missing:
            print(f"  - {message}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table + "\n")
        print(f"\nSaved table to {args.output}")


if __name__ == "__main__":
    main()
