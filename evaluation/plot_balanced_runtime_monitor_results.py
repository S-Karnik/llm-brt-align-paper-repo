import argparse
import json
import os
import glob
from statistics import mean, stdev
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import directories

ROBERTA_BASELINE_RESULTS = {
    "Qwen2-1.5B": {"BeaverTails": 0.580, "Real Toxicity": 0.664, "Ultra Safety": 0.205},
    "Llama2-7b": {"BeaverTails": 0.656, "Real Toxicity": 0.684, "Ultra Safety": 0.468},
    "Ministral-8B-Instruct-2410": {"BeaverTails": 0.639, "Real Toxicity": 0.660, "Ultra Safety": 0.277},
    "Falcon-7B": {"BeaverTails": 0.543, "Real Toxicity": 0.655, "Ultra Safety": 0.344},
    "gpt-oss-20b": {"BeaverTails": 0.348, "Real Toxicity": 0.435, "Ultra Safety": 0.174},
}

def mean_std(values):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)

def compute_balanced_f1(safe_acc, unsafe_acc):
    # assume 50/50 class balance, N per class
    N = 1.0
    TP = unsafe_acc * N
    FN = N - TP
    TN = safe_acc * N
    FP = N - TN

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)

def load_results(results_root, model_name, aggregate_datasets):
    parsed_model_name = model_name.replace("/", "_") + ".json"
    json_paths = glob.glob(os.path.join(results_root, "seed_*", parsed_model_name))
    if not json_paths:
        raise FileNotFoundError(f"No JSONs found for {parsed_model_name} under {results_root}")

    per_seed = {}
    for jp in json_paths:
        seed = os.path.basename(os.path.dirname(jp))
        with open(jp, "r") as f:
            data = json.load(f)

        for method, datasets in data.items():
            if aggregate_datasets:
                f1_vals = []
                for dataset_metrics in datasets.values():
                    f1_score = compute_balanced_f1(
                        safe_acc=dataset_metrics["total_safe_classification_rate"],
                        unsafe_acc=dataset_metrics["total_unsafe_classification_rate"],
                    )
                    f1_vals.append(f1_score)
                avg_f1 = mean(f1_vals) if f1_vals else 0.0
                per_seed.setdefault((method, "ALL"), {})[seed] = {"f1_unsafe": avg_f1}
            else:
                for dataset_name, dataset_metrics in datasets.items():
                    f1_score = compute_balanced_f1(
                        safe_acc=dataset_metrics["total_safe_classification_rate"],
                        unsafe_acc=dataset_metrics["total_unsafe_classification_rate"],
                    )
                    dataset_metrics["f1_unsafe"] = f1_score
                    per_seed.setdefault((method, dataset_name), {})[seed] = {
                        "f1_unsafe": dataset_metrics.get("f1_unsafe", 0.0)
                    }
    return per_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        type=str,
        default=os.path.join(directories.DATA_CACHE_DIR, "final_runtime_monitor_results"),
    )
    parser.add_argument("--aggregate_datasets", action="store_true")
    parser.add_argument("--dataset_filter", type=str, default=None,
                        help="Restrict to a single dataset (e.g. 'BeaverTails', 'Real Toxicity', 'Ultra Safety')")
    parser.add_argument("--save_path", type=str, default="balanced_f1_scores.png")
    args = parser.parse_args()

    model_name_map = {
        "Qwen/Qwen2-1.5B": "Qwen2-1.5B",
        "meta-llama/Llama-2-7b-hf": "Llama2-7b",
        "mistralai/Ministral-8B-Instruct-2410": "Ministral-8B-Instruct-2410",
        "tiiuae/falcon-7b-instruct": "Falcon-7B",
        "openai/gpt-oss-20b": "gpt-oss-20b",
    }

    ordered_models = ["Qwen2-1.5B", "Llama2-7b", "Ministral-8B-Instruct-2410", "Falcon-7B", "gpt-oss-20b"]
    method_order = ["SaP", "Re-Control", "BRT", "Sample-BRT"]

    records = []
    for raw_model, display_model in model_name_map.items():
        try:
            per_seed = load_results(args.results_root, raw_model, args.aggregate_datasets)
        except FileNotFoundError:
            continue
        for (method, dataset), seeds_dict in per_seed.items():
            # Apply dataset filter if given
            if args.dataset_filter and dataset != args.dataset_filter:
                continue
            vals = [m["f1_unsafe"] for m in seeds_dict.values()]
            mu, sd = mean_std(vals)
            records.append({"Method": method, "Model": display_model,
                            "F1_mean": mu, "F1_std": sd})

    # Add EOS classifier
    for model, datasets in ROBERTA_BASELINE_RESULTS.items():
        avg_f1 = sum(datasets.values()) / len(datasets)
        records.append({
            "Method": "EOS Classifier (RoBERTa)",
            "Model": model,
            "F1_mean": avg_f1,
            "F1_std": 0.0,
        })

    df = pd.DataFrame(records)
    df["Model"] = pd.Categorical(df["Model"], categories=ordered_models, ordered=True)

    method_colors = {
        "BRT": "#d46322",
        "Sample-BRT": "#ff9500",
        "Re-Control": "#9B9059",
        "SaP": "#806363",
    }

    # sns.set_theme(style="whitegrid", font_scale=1.4)
    # sns.set_style("whitegrid", {
    #     "grid.color": ".9",   # light gray
    #     "grid.linestyle": "-",
    #     "grid.linewidth": 0.7
    # })
    plt.figure(figsize=(14, 7))

    ax = sns.barplot(
        data=df,
        y="Model", x="F1_mean", hue="Method",
        hue_order=method_order,
        palette=method_colors,
        orient="h",
        errorbar=None
    )
    ax.set_xlim(0, 1.0)
    ax.set_yticklabels([])
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels([])
    ax.grid(False)
    if ax.legend_ is not None:
        ax.legend_.remove()

    # Add error bars + numeric labels
    for container, method in zip(ax.containers, method_order):
        subset = df[df["Method"] == method]
        for patch, (_, row) in zip(container, subset.iterrows()):
            y = patch.get_y() + patch.get_height() / 2

            # Draw error bar
            if method != "EOS Classifier (RoBERTa)" and row["F1_std"] > 0:
                print(row["Model"], row["Method"], row["F1_mean"], row["F1_std"])
                ax.errorbar(
                    x=row["F1_mean"], y=y,
                    xerr=row["F1_std"],
                    fmt="none", c="black", capsize=4, lw=1.5
                )

            # Add text label with mean ± std
            # ax.text(
            #     row["F1_mean"] + 0.01,
            #     y,
            #     f"{row['F1_mean']:.3f} ± {row['F1_std']:.3f}",
            #     va="center",
            #     ha="left",
            #     fontsize=12
            # )


    ax.set_xlabel("", fontsize=24)
    ax.set_ylabel("", labelpad=20)

    plt.tight_layout()
    plt.savefig(args.save_path, dpi=300)
    print(f"Saved plot to {args.save_path}")

if __name__ == "__main__":
    main()
