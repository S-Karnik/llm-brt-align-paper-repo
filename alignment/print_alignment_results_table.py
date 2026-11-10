import argparse
import os
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List
from tabulate import tabulate

import directories

DATASET_PAIRS = [("beavertails", "330k_test"),
                 ("real_toxicity", "train"),
                 ("ultrasafety", "train")]
METHOD_NAMES = ["No Alignment", "SaP", "Re-Control", "Sample-BRT", "BRT"]
METRICS = ["average_safety_rate", "average_cosine_similarity", "average_word_diversity"]

# Constants
RESULTS_ROOT = os.path.join(directories.DATA_CACHE_DIR, "final_alignment_results_only_unsafe")
SEEDS = [41, 42, 43, 44, 45]

def safe_mean_std(values: List[float]):
    """Compute mean and std, handling empty lists."""
    if not values:
        return None, None
    arr = np.array(values, dtype=float)
    if len(arr) == 1:
        return float(arr.mean()), None
    return float(arr.mean()), float(arr.std(ddof=1))

def load_seed_json(base_dir: str, seed: int, model_name: str) -> Dict:
    parsed_model_name = model_name.replace("/", "_")
    json_path = os.path.join(base_dir, f"seed_{seed}", f"{parsed_model_name}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing JSON for seed {seed}: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)

def aggregate_results(base_dir: str, seeds: List[int], model_name: str, aggregate_datasets: bool):
    if not aggregate_datasets:
        # nested dict: dataset -> method -> metric -> list of values across seeds
        all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for seed in seeds:
            seed_data = load_seed_json(base_dir, seed, model_name)
            for dataset, methods in seed_data.items():
                for method, metrics in methods.items():
                    for metric_name in METRICS:
                        val = metrics.get(metric_name, None)
                        if val is not None:
                            all_results[dataset][method][metric_name].append(val)
        aggregated = defaultdict(lambda: defaultdict(dict))
        for dataset, methods in all_results.items():
            for method, metrics in methods.items():
                for metric_name, values in metrics.items():
                    mean, std = safe_mean_std(values)
                    aggregated[dataset][method][metric_name] = (mean, std)
        return aggregated
    else:
        # aggregate across datasets first
        all_results = defaultdict(lambda: defaultdict(list))  # method -> metric -> list of per-seed averages
        for seed in seeds:
            seed_data = load_seed_json(base_dir, seed, model_name)
            for method in METHOD_NAMES:
                for metric in METRICS:
                    per_dataset_vals = []
                    for dataset in seed_data.keys():
                        val = seed_data[dataset].get(method, {}).get(metric, None)
                        if val is not None:
                            per_dataset_vals.append(val)
                    if per_dataset_vals:
                        avg_val = np.mean(per_dataset_vals)
                        all_results[method][metric].append(avg_val)

        aggregated = defaultdict(dict)  # method -> metric -> (mean, std)
        for method, metrics in all_results.items():
            for metric, values in metrics.items():
                mean, std = safe_mean_std(values)
                aggregated[method][metric] = (mean, std)
        return aggregated

def format_val(mean, std):
    if mean is None:
        return "N/A"
    # Handle the case where std is None or not a valid float
    if std is None:
        return f"{mean:.3f}"
    try:
        if np.isnan(std):
            return f"{mean:.3f}"
    except TypeError:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def print_table(aggregated, aggregate_datasets: bool):
    if not aggregate_datasets:
        for dataset, methods in aggregated.items():
            print(f"\n=== Dataset: {dataset} ===\n")
            table_data = []
            for method in METHOD_NAMES:
                row = [method]
                if method not in methods:
                    row.extend(["-"] * len(METRICS))
                else:
                    for metric in METRICS:
                        mean, std = methods[method].get(metric, (None, None))
                        row.append(format_val(mean, std))
                table_data.append(row)
            print(tabulate(table_data, headers=["Method"] + METRICS, tablefmt="github"))
    else:
        print(f"\n=== Aggregated Across Datasets ===\n")
        table_data = []
        for method in METHOD_NAMES:
            row = [method]
            if method not in aggregated:
                row.extend(["-"] * len(METRICS))
            else:
                for metric in METRICS:
                    mean, std = aggregated[method].get(metric, (None, None))
                    row.append(format_val(mean, std))
            table_data.append(row)
        print(tabulate(table_data, headers=["Method"] + METRICS, tablefmt="github"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, action="append", required=True)
    p.add_argument("--results_root", type=str, default=RESULTS_ROOT)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--aggregate_datasets", action="store_true",
                   help="If set, compute averages across datasets instead of per-dataset results.")
    args = p.parse_args()

    for model_name in args.model_name:
        print(f"\n##############################")
        print(f"# Model: {model_name}")
        print(f"##############################")
        aggregated = aggregate_results(args.results_root, args.seeds, model_name, args.aggregate_datasets)
        print_table(aggregated, args.aggregate_datasets)
