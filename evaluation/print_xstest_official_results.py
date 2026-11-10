#!/usr/bin/env python3
"""Aggregate cached official XSTest labels across training seeds."""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_NAMES = {
    "brt": "RL-BRT-Align",
    "sample_brt": "Sample-BRT-Align",
    "recontrol": "Re-Control",
    "sap": "SaP",
}


def identify_run(source_file, model_name):
    path = Path(source_file)
    parts = path.parts
    seed_match = re.search(r"/seed_(\d+)/", path.as_posix())
    if seed_match is None or model_name.replace("/", "_") not in parts:
        return None

    if path.parent.name == "base_llm":
        method = "Base LLM"
    else:
        try:
            xstest_index = parts.index("XSTest")
            method_key = parts[xstest_index + 2]
        except (ValueError, IndexError):
            return None
        method = METHOD_NAMES.get(method_key, method_key)

    return int(seed_match.group(1)), method


def compute_metrics(frame):
    unsafe = frame["type"].astype(str).str.startswith("contrast_")
    refusal = frame["gpt4_label"].isin(
        ["2_full_refusal", "3_partial_refusal"]
    )
    safe = ~unsafe
    return {
        "Safe full refusal": (frame.loc[safe, "gpt4_label"] == "2_full_refusal").mean(),
        "Safe partial refusal": (frame.loc[safe, "gpt4_label"] == "3_partial_refusal").mean(),
        "Safe any refusal": refusal[safe].mean(),
        "Unsafe any refusal": refusal[unsafe].mean(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()

    runs = []
    seen_sources = set()
    for csv_path in sorted(args.results_dir.glob("*_official_gpt.csv")):
        frame = pd.read_csv(csv_path)
        if frame.empty or "source_response_file" not in frame:
            continue
        source_file = str(frame["source_response_file"].iloc[0])
        if source_file in seen_sources:
            continue
        identified = identify_run(source_file, args.model_name)
        if identified is None:
            continue
        seed, method = identified
        seen_sources.add(source_file)
        runs.append({"seed": seed, "method": method, **compute_metrics(frame)})

    if not runs:
        parser.error(f"No cached results found for {args.model_name}")

    run_frame = pd.DataFrame(runs)
    metric_names = [
        "Safe full refusal",
        "Safe partial refusal",
        "Safe any refusal",
        "Unsafe any refusal",
    ]

    print(f"XSTest results for {args.model_name} (mean +/- std across seeds)\n")
    print("| Method | Seeds | " + " | ".join(metric_names) + " |")
    print("|---|---:|" + "---:|" * len(metric_names))
    for method, group in run_frame.groupby("method", sort=True):
        values = []
        for metric in metric_names:
            mean = group[metric].mean()
            std = group[metric].std(ddof=0)
            values.append(f"{mean:.4f} +/- {std:.4f}")
        seeds = ",".join(str(seed) for seed in sorted(group["seed"].unique()))
        print(f"| {method} | {seeds} | " + " | ".join(values) + " |")


if __name__ == "__main__":
    main()
