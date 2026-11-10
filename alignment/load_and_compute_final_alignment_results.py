import argparse
import os
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import directories

DATASET_PAIRS = [("beavertails", "330k_test_only_unsafe"), ("real_toxicity", "train_only_unsafe"), ("ultrasafety", "train_only_unsafe")]
METHOD_NAMES = ["No Alignment", "SaP", "Re-Control", "Sample-BRT", "BRT"]


def _method_min_mtime(method: str, min_mtime: Optional[float], methods_with_min_mtime) -> Optional[float]:
    if min_mtime is None:
        return None
    if not methods_with_min_mtime:
        return min_mtime
    return min_mtime if method in methods_with_min_mtime else None

def _parse_min_mtime(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    return datetime.fromisoformat(value).timestamp()


def _check_mtime(path: Path, min_mtime: Optional[float]) -> None:
    if min_mtime is not None and path.stat().st_mtime < min_mtime:
        observed = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
        required = datetime.fromtimestamp(min_mtime).isoformat(timespec="seconds")
        raise FileNotFoundError(f"Result file is older than min_mtime ({observed} < {required}): {path}")


def _load_json_list(path: Path) -> List[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]


def _has_result_files(path: Path, filename: str) -> bool:
    return (
        (path / filename).exists()
        or any(path.glob("qualitative_prompt_output_shard_*_of_*_shards.json"))
    )


def _result_subdirs(path: Path, filename: str):
    subdirs = [d for d in path.iterdir() if d.is_dir() and "calibration" not in d.name]
    result_subdirs = [d for d in subdirs if _has_result_files(d, filename)]

    if path.name == "brt_least_restrictive":
        result_subdirs = [
            d for d in result_subdirs
            if d.name.startswith("brt_least_restrictive_")
        ]
    elif path.name == "sap":
        result_subdirs = [
            d for d in result_subdirs
            if d.name.startswith("sap_lambda_weight_")
        ]
    elif path.name == "recontrol":
        result_subdirs = [
            d for d in result_subdirs
            if d.name.startswith("recontrol_num_updates_")
        ]

    return subdirs, result_subdirs


def _load_shard_files(path: Path, min_mtime: Optional[float] = None) -> Optional[List[dict]]:
    candidate_shard_files = sorted(path.glob("qualitative_prompt_output_shard_*_of_*_shards.json"))
    if not candidate_shard_files:
        return None

    shard_pattern = re.compile(r"qualitative_prompt_output_shard_(\d+)_of_(\d+)_shards\.json$")
    shard_indices = {}
    stale_shard_files = []
    for shard_file in candidate_shard_files:
        match = shard_pattern.match(shard_file.name)
        if not match:
            continue
        if min_mtime is not None and shard_file.stat().st_mtime < min_mtime:
            stale_shard_files.append(shard_file)
            continue
        shard_idx, num_shards = map(int, match.groups())
        shard_indices.setdefault(num_shards, {})[shard_idx] = shard_file

    if not shard_indices:
        if stale_shard_files:
            oldest = min(stale_shard_files, key=lambda shard_file: shard_file.stat().st_mtime)
            _check_mtime(oldest, min_mtime)
        return None

    if len(shard_indices) > 1:
        summary = {
            num_shards: sorted(shards)
            for num_shards, shards in sorted(shard_indices.items())
        }
        raise RuntimeError(f"Ambiguous fresh shard layouts under {path}: {summary}")

    num_shards, shards_by_idx = next(iter(shard_indices.items()))
    expected = set(range(num_shards))
    missing = sorted(expected - set(shards_by_idx))
    if missing:
        raise FileNotFoundError(f"Missing shard(s) {missing} of {num_shards} under {path}")

    merged = []
    shard_files = [shards_by_idx[idx] for idx in range(num_shards)]
    for shard_file in shard_files:
        merged.extend(_load_json_list(shard_file))
    print(f"Loaded {len(merged)} records from {len(shard_files)} shard files under {path}")
    return merged


def load_results(
    base_dir: str,
    seed: int,
    model: str,
    dataset: str,
    split: str,
    method: str,
    filename: str = "combined.json",
    max_depth: int = 2,
    prefer_shards: bool = False,
    min_mtime: Optional[float] = None,
) -> List[dict]:
    """
    Loads the merged JSON results for a given configuration.

    Search strategy:
    - Look in method directory for combined.json.
    - If not found, load shard JSON files directly from the same directory.
    - If not found, descend up to `max_depth` levels if each level has exactly one subdir.
    - Raise error if multiple subdirs exist (ambiguous).
    """
    path = Path(base_dir) / f"seed_{seed}" / model / dataset / split / method

    for depth in range(max_depth + 1):
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        subdirs, result_subdirs = _result_subdirs(path, filename)
        if result_subdirs:
            if len(result_subdirs) > 1:
                raise RuntimeError(
                    f"Ambiguous: multiple result subdirs under {path}: {[d.name for d in result_subdirs]}"
                )
            print(f"Descending into result subdir {result_subdirs[0]}")
            path = result_subdirs[0]
            continue

        if prefer_shards:
            shard_data = _load_shard_files(path, min_mtime=min_mtime)
            if shard_data is not None:
                return shard_data

        candidate = path / filename
        print(f"Looking for {candidate}")
        if candidate.exists():
            _check_mtime(candidate, min_mtime)
            print(f"candidate = {candidate}")
            return _load_json_list(candidate)

        shard_data = _load_shard_files(path, min_mtime=min_mtime)
        if shard_data is not None:
            return shard_data

        # descend if exactly one subdir
        if len(subdirs) == 1:
            path = subdirs[0]
        elif len(subdirs) == 0:
            raise FileNotFoundError(f"No {filename} found under {path}")
        else:
            found_multiple_results = 0
            for d in subdirs:
                if _has_result_files(d, filename):
                    path = d
                    found_multiple_results += 1
            if found_multiple_results > 1:
                raise RuntimeError(
                    f"Ambiguous: multiple subdirs under {path}: {[d.name for d in subdirs]}"
                )

    raise FileNotFoundError(
        f"Reached max depth={max_depth} without finding {filename} for {method}"
    )

if __name__ == "__main__":
    from alignment import analyze_alignment_results

    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2-1.5B")
    p.add_argument("--seed", type=int)
    p.add_argument("--allow_missing", action="store_true")
    p.add_argument("--prefer_shards", action="store_true")
    p.add_argument("--force_recompute", action="store_true")
    p.add_argument(
        "--full_sample_size",
        type=int,
        default=None,
        help=(
            "Evaluate full-dataset sampled runs instead of the default only-unsafe runs. "
            "For example, 1000 selects split directories such as "
            "330k_test_sampled_1000_sample_seed_2026."
        ),
    )
    p.add_argument(
        "--sample_dataset_seed",
        type=int,
        default=2026,
        help="Sampling seed used in full-dataset sampled split directory names.",
    )
    p.add_argument(
        "--alignment_results_root",
        type=str,
        default=os.path.join(directories.DATA_CACHE_DIR, "alignment_results"),
        help="Root directory containing raw alignment result folders.",
    )
    p.add_argument(
        "--final_results_root",
        type=str,
        default=os.path.join(directories.DATA_CACHE_DIR, "final_alignment_results_only_unsafe"),
        help="Root directory where computed final metric JSONs are written.",
    )
    p.add_argument("--min_mtime", type=str, default=None,
                   help="Only use result files newer than this ISO timestamp or epoch seconds.")
    p.add_argument(
        "--min_mtime_method",
        action="append",
        default=[],
        choices=["sap", "recontrol", "sample_brt", "brt"],
        help="Apply --min_mtime only to this method. Repeatable. Default: apply to all methods.",
    )

    args = p.parse_args()
    min_mtime = _parse_min_mtime(args.min_mtime)
    results_root = args.alignment_results_root
    dataset_pairs = DATASET_PAIRS
    if args.full_sample_size is not None:
        sample_suffix = f"_sampled_{args.full_sample_size}_sample_seed_{args.sample_dataset_seed}"
        dataset_pairs = [
            ("beavertails", f"330k_test{sample_suffix}"),
            ("real_toxicity", f"train{sample_suffix}"),
            ("ultrasafety", f"train{sample_suffix}"),
        ]

    parsed_model_name = args.model_name.replace('/', '_')
    
    save_dir = f"{args.final_results_root}/seed_{args.seed}"
    os.makedirs(save_dir, exist_ok=True)

    json_path = os.path.join(save_dir, f"{parsed_model_name}.json")

    results_data = defaultdict(dict)

    # To skip the methods we've already computed
    completed = {d: {m: False for m in METHOD_NAMES} for d, _ in dataset_pairs}
    if args.force_recompute:
        print(f"Ignoring cached final metrics because --force_recompute was set: {json_path}")
    elif os.path.exists(json_path) and min_mtime is not None and Path(json_path).stat().st_mtime < min_mtime:
        observed = datetime.fromtimestamp(Path(json_path).stat().st_mtime).isoformat(timespec="seconds")
        required = datetime.fromtimestamp(min_mtime).isoformat(timespec="seconds")
        print(f"Ignoring cached final metrics older than min_mtime ({observed} < {required}): {json_path}")
    elif os.path.exists(json_path):
        with open(json_path, "r") as f:
            loaded_json = json.load(f)

        for dataset_pair in dataset_pairs:
            dataset_name = dataset_pair[0]
            if dataset_pair[0] in loaded_json:
                for method in METHOD_NAMES:
                    if method in loaded_json[dataset_name] and "average_safety_rate" in loaded_json[dataset_name][method]:
                        completed[dataset_name][method] = True
                        results_data[dataset_name][method] = loaded_json[dataset_name][method]

    for dataset_name, split in dataset_pairs:
        print(f"Generating results for model_name = {args.model_name}, dataset_name = {dataset_name}")

        # SaP (direct combined.json at method level)

        if not completed[dataset_name]["No Alignment"] or not completed[dataset_name]["SaP"]:
            try:
                sap_data = load_results(
                    results_root, seed=args.seed,
                    model=parsed_model_name,
                    dataset=dataset_name, split=split, method="sap",
                    prefer_shards=args.prefer_shards,
                    min_mtime=_method_min_mtime("sap", min_mtime, args.min_mtime_method),
                )
                print("SaP Results")
                sap_alignment_metrics, baseline_metrics = analyze_alignment_results.compute_alignment_metrics(sap_data, compute_baseline_metrics=True)

                results_data[dataset_name]["No Alignment"] = {
                    "average_safety_rate": baseline_metrics.average_safety,
                    "average_cosine_similarity": baseline_metrics.average_cosine_similarity,
                    "average_word_diversity": baseline_metrics.average_word_diversity
                }

                results_data[dataset_name]["SaP"] = {
                    "average_safety_rate": sap_alignment_metrics.average_safety,
                    "average_cosine_similarity": sap_alignment_metrics.average_cosine_similarity,
                    "average_word_diversity": sap_alignment_metrics.average_word_diversity
                }
            except (FileNotFoundError, RuntimeError) as e:
                if not args.allow_missing:
                    raise
                print(f"Skipping SaP for {dataset_name}: {e}")

        if ("Re-Control" in completed[dataset_name]) and not completed[dataset_name]["Re-Control"]:
            # ReControl (combined.json inside 1 subdir)
            try:
                recontrol_data = load_results(
                    results_root, seed=args.seed,
                    model=parsed_model_name,
                    dataset=dataset_name, split=split, method="recontrol",
                    prefer_shards=args.prefer_shards,
                    min_mtime=_method_min_mtime("recontrol", min_mtime, args.min_mtime_method),
                )
                print("Re-Control Results")
                recontrol_metrics, _ = analyze_alignment_results.compute_alignment_metrics(recontrol_data, compute_baseline_metrics=False)

                results_data[dataset_name]["Re-Control"] = {
                    "average_safety_rate": recontrol_metrics.average_safety,
                    "average_cosine_similarity": recontrol_metrics.average_cosine_similarity,
                    "average_word_diversity": recontrol_metrics.average_word_diversity
                }
            except (FileNotFoundError, RuntimeError) as e:
                if not args.allow_missing:
                    raise
                print(f"Skipping Re-Control for {dataset_name}: {e}")


        if ("Sample-BRT" in completed[dataset_name]) and not completed[dataset_name]["Sample-BRT"]:
            # Sample-BRT (combined.json inside 2 subdirs)
            try:
                sample_brt_data = load_results(
                    results_root, seed=args.seed,
                    model=parsed_model_name,
                    dataset=dataset_name, split=split, method="sample_brt",
                    prefer_shards=args.prefer_shards,
                    min_mtime=_method_min_mtime("sample_brt", min_mtime, args.min_mtime_method),
                )

                print("Sample BRT Results")
                sample_brt_metrics, _ = analyze_alignment_results.compute_alignment_metrics(sample_brt_data, compute_baseline_metrics=False)

                results_data[dataset_name]["Sample-BRT"] = {
                    "average_safety_rate": sample_brt_metrics.average_safety,
                    "average_cosine_similarity": sample_brt_metrics.average_cosine_similarity,
                    "average_word_diversity": sample_brt_metrics.average_word_diversity
                }
            except (FileNotFoundError, RuntimeError) as e:
                if not args.allow_missing:
                    raise
                print(f"Skipping Sample-BRT for {dataset_name}: {e}")
        
        if ("BRT" in completed[dataset_name]) and not completed[dataset_name]["BRT"]:
            # Sample-BRT (combined.json inside 2 subdirs)
            try:
                sample_brt_data = load_results(
                    results_root, seed=args.seed,
                    model=parsed_model_name,
                    dataset=dataset_name, split=split, method="brt",
                    prefer_shards=args.prefer_shards,
                    min_mtime=_method_min_mtime("brt", min_mtime, args.min_mtime_method),
                )

                print("BRT Results")
                sample_brt_metrics, _ = analyze_alignment_results.compute_alignment_metrics(sample_brt_data, compute_baseline_metrics=False)

                results_data[dataset_name]["BRT"] = {
                    "average_safety_rate": sample_brt_metrics.average_safety,
                    "average_cosine_similarity": sample_brt_metrics.average_cosine_similarity,
                    "average_word_diversity": sample_brt_metrics.average_word_diversity
                }
            except (FileNotFoundError, RuntimeError) as e:
                if not args.allow_missing:
                    raise
                print(f"Skipping BRT for {dataset_name}: {e}")

    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)
