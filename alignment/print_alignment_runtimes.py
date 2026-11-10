import os
import json
import numpy as np
import directories

def main():
    MODELS = [
        "Qwen/Qwen2-1.5B",
        "meta-llama/Llama-2-7b-hf",
        "mistralai/Ministral-8B-Instruct-2410",
        "tiiuae/falcon-7b-instruct",
        "openai/gpt-oss-20b"
    ]
    DATASETS = ["beavertails", "real_toxicity", "ultrasafety"]
    SPLITS = {
        "beavertails": "330k_test_only_unsafe_sampled_1000000000",
        "real_toxicity": "train_only_unsafe_sampled_1000000000",
        "ultrasafety": "train_only_unsafe_sampled_1000000000"
    }
    METHODS = ["sap", "recontrol", "sample_brt", "brt"]
    ALIGNMENT_METHODS = {
        "brt": "brt_least_restrictive",
        "sample_brt": "brt_least_restrictive",
        "recontrol": "",
        "sap": ""
    }
    SEEDS = [41, 42, 43, 44, 45]

    RESULTS_DIR = os.path.join(directories.DATA_CACHE_DIR, "alignment_results")

    # Collect all results in a nested dictionary: results[model][method][dataset][seed] = [avg_times]
    results = {model: {method: {dataset: {seed: [] for seed in SEEDS} for dataset in DATASETS} for method in METHODS} for model in MODELS}

    for model in MODELS:
        processed_model = model.replace('/', '_')
        for method in METHODS:
            for dataset in DATASETS:
                split_name = SPLITS[dataset]
                for seed in SEEDS:
                    base_dir = os.path.join(RESULTS_DIR, f"seed_{seed}", processed_model, dataset, split_name)
                    method_dir = None
                    for root, dirs, files in os.walk(base_dir):
                        for d in dirs:
                            if d == method or d.endswith(f"/{method}"):
                                method_dir = os.path.join(root, d)
                                break
                        if method_dir:
                            break
                    if not method_dir:
                        continue
                    method_dir = os.path.join(method_dir, ALIGNMENT_METHODS[method]) if ALIGNMENT_METHODS[method] else method_dir
                    final_dir = method_dir
                    while True:
                        subdirs = [d for d in os.listdir(final_dir) if os.path.isdir(os.path.join(final_dir, d))]
                        if not subdirs:
                            break
                        final_dir = os.path.join(final_dir, subdirs[0])
                    for fname in os.listdir(final_dir):
                        if fname.startswith("time_info") and fname.endswith(".json"):
                            fpath = os.path.join(final_dir, fname)
                            try:
                                with open(fpath, "r") as f:
                                    info = json.load(f)
                                results[model][method][dataset][seed].append(info.get("avg_time_per_prompt", 0.0))
                            except Exception:
                                continue

    # Print per-dataset/method/model table
    print(f"{'Model':<40} {'Dataset':<15} {'Method':<15} {'Mean (s)':<12} {'Std (s)':<12}")
    print("-" * 94)
    for model in MODELS:
        for dataset in DATASETS:
            for method in METHODS:
                all_times = []
                for seed in SEEDS:
                    all_times.extend(results[model][method][dataset][seed])
                if all_times:
                    mean = np.mean(all_times)
                    std = np.std(all_times)
                    print(f"{model:<40} {dataset:<15} {method:<15} {mean:<12.4f} {std:<12.4f}")
                else:
                    print(f"{model:<40} {dataset:<15} {method:<15} {'N/A':<12} {'N/A':<12}")

    # Print summary: Mean and Std of cross-dataset averages across seeds
    print("\nSummary: Mean and Std of cross-dataset averages across seeds")
    print(f"{'Model':<40} {'Method':<15} {'Mean (s)':<12} {'Std (s)':<12}")
    print("-" * 80)
    for model in MODELS:
        for method in METHODS:
            seed_avgs = []
            for seed in SEEDS:
                dataset_means = []
                for dataset in DATASETS:
                    times = results[model][method][dataset][seed]
                    if times:
                        dataset_means.append(np.mean(times))
                if dataset_means:
                    seed_avgs.append(np.mean(dataset_means))
            if seed_avgs:
                mean = np.mean(seed_avgs)
                std = np.std(seed_avgs)
                print(f"{model:<40} {method:<15} {mean:<12.4f} {std:<12.4f}")
            else:
                print(f"{model:<40} {method:<15} {'N/A':<12} {'N/A':<12}")

if __name__ == "__main__":
    main()
