#!/usr/bin/env python3
"""Print final Gemini safety and coherence results across separate seed runs."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from tabulate import tabulate

import directories

DEFAULT_ROOT = Path(directories.DATA_CACHE_DIR)
DEFAULT_MODELS = [
    "Qwen/Qwen2-1.5B",
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Ministral-8B-Instruct-2410",
    "tiiuae/falcon-7b-instruct",
    "openai/gpt-oss-20b",
]
DEFAULT_SEEDS = [41, 42, 43, 44, 45]
METHODS = ["sap", "recontrol", "sample_brt", "brt"]
METHOD_LABELS = {
    "sap": "SaP",
    "recontrol": "Re-Control",
    "sample_brt": "Sample-BRT",
    "brt": "BRT",
}


def _mean_std(values):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def _work_dirs(root, model, seed, target_prompts, gpt_ultrasafety_prompts):
    model_tag = model.replace("/", "_")
    prefix = root / f"gemini_judge_{model_tag}_seed{seed}"
    if model == "openai/gpt-oss-20b":
        return [
            Path(f"{prefix}_n{target_prompts}_main"),
            Path(f"{prefix}_n{gpt_ultrasafety_prompts}_ultrasafety"),
        ]
    return [Path(f"{prefix}_n{target_prompts}")]


def _load_cells(args):
    cells = {}
    missing = []
    for model in args.models:
        for seed in args.seeds:
            for work_dir in _work_dirs(
                args.root,
                model,
                seed,
                args.target_prompts,
                args.gpt_ultrasafety_prompts,
            ):
                summary_path = work_dir / "summary.json"
                if not summary_path.exists():
                    missing.append(summary_path)
                    continue
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                for cell in summary["per_seed_dataset"].values():
                    key = (
                        cell["model"],
                        cell["method"],
                        int(cell["seed"]),
                        cell["dataset"],
                    )
                    if key in cells:
                        raise RuntimeError(f"Duplicate result cell {key} from {summary_path}")
                    cells[key] = cell

    if missing and not args.allow_missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing final Gemini summaries:\n{formatted}")
    if missing:
        print(f"Warning: skipped {len(missing)} missing summary files.")
    return cells


def _aggregate(cells, models, seeds):
    rows = []
    for model in models:
        for method in METHODS:
            per_seed_safety = []
            per_seed_coherence = []
            total_judgments = 0
            for seed in seeds:
                seed_cells = [
                    cell
                    for (cell_model, cell_method, cell_seed, _), cell in cells.items()
                    if cell_model == model
                    and cell_method == method
                    and cell_seed == seed
                ]
                if not seed_cells:
                    continue
                per_seed_safety.append(
                    statistics.mean(cell["safety_score"] for cell in seed_cells)
                )
                per_seed_coherence.append(
                    statistics.mean(
                        cell["generated_coherence_score"] for cell in seed_cells
                    )
                )
                total_judgments += sum(cell["n"] for cell in seed_cells)

            if not per_seed_safety:
                continue
            safety_mean, safety_std = _mean_std(per_seed_safety)
            coherence_mean, coherence_std = _mean_std(per_seed_coherence)
            rows.append([
                model,
                METHOD_LABELS[method],
                f"{safety_mean:.3f} +/- {safety_std:.3f}",
                f"{coherence_mean:.3f} +/- {coherence_std:.3f}",
                len(per_seed_safety),
                total_judgments,
            ])
    return rows


def _aggregate_win_tie_loss(cells, models, seeds):
    rows = []
    for model in models:
        for method in METHODS:
            per_seed_wins = []
            per_seed_ties = []
            per_seed_losses = []
            total_judgments = 0
            for seed in seeds:
                seed_cells = [
                    cell
                    for (cell_model, cell_method, cell_seed, _), cell in cells.items()
                    if cell_model == model
                    and cell_method == method
                    and cell_seed == seed
                ]
                seed_n = sum(cell["n"] for cell in seed_cells)
                if not seed_n:
                    continue
                per_seed_wins.append(sum(
                    cell["generated_safer_rate"] * cell["n"] for cell in seed_cells
                ) / seed_n)
                per_seed_ties.append(sum(
                    cell["same_safety_rate"] * cell["n"] for cell in seed_cells
                ) / seed_n)
                per_seed_losses.append(sum(
                    cell["baseline_safer_rate"] * cell["n"] for cell in seed_cells
                ) / seed_n)
                total_judgments += seed_n

            if not per_seed_wins:
                continue
            win_mean, win_std = _mean_std(per_seed_wins)
            tie_mean, tie_std = _mean_std(per_seed_ties)
            loss_mean, loss_std = _mean_std(per_seed_losses)
            per_seed_margins = [
                win - loss for win, loss in zip(per_seed_wins, per_seed_losses)
            ]
            margin_mean, margin_std = _mean_std(per_seed_margins)
            rows.append([
                model,
                METHOD_LABELS[method],
                f"{100 * win_mean:.1f} +/- {100 * win_std:.1f}",
                f"{100 * tie_mean:.1f} +/- {100 * tie_std:.1f}",
                f"{100 * loss_mean:.1f} +/- {100 * loss_std:.1f}",
                f"{100 * margin_mean:.1f} +/- {100 * margin_std:.1f}",
                len(per_seed_wins),
                total_judgments,
            ])
    return rows


def _print_per_dataset(cells, models):
    grouped = defaultdict(lambda: {"safety": [], "coherence": [], "n": 0})
    for (model, method, _, dataset), cell in cells.items():
        group = grouped[(model, method, dataset)]
        group["safety"].append(cell["safety_score"])
        group["coherence"].append(cell["generated_coherence_score"])
        group["n"] += cell["n"]

    rows = []
    for model in models:
        datasets = sorted({key[2] for key in grouped if key[0] == model})
        for dataset in datasets:
            for method in METHODS:
                group = grouped.get((model, method, dataset))
                if not group:
                    continue
                safety_mean, safety_std = _mean_std(group["safety"])
                coherence_mean, coherence_std = _mean_std(group["coherence"])
                rows.append([
                    model,
                    dataset,
                    METHOD_LABELS[method],
                    f"{safety_mean:.3f} +/- {safety_std:.3f}",
                    f"{coherence_mean:.3f} +/- {coherence_std:.3f}",
                    len(group["safety"]),
                    group["n"],
                ])
    print(tabulate(
        rows,
        headers=["LLM", "Dataset", "Method", "Safety (1-3)", "Coherence (1-5)", "Seeds", "N"],
        tablefmt="github",
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--target_prompts", type=int, default=30)
    parser.add_argument("--gpt_ultrasafety_prompts", type=int, default=8)
    parser.add_argument("--per_dataset", action="store_true")
    parser.add_argument(
        "--win_tie_loss",
        action="store_true",
        help="Print generated-safer/tie/baseline-safer percentages instead of mean scores.",
    )
    parser.add_argument("--allow_missing", action="store_true")
    args = parser.parse_args()

    cells = _load_cells(args)
    if args.per_dataset:
        _print_per_dataset(cells, args.models)
        return

    if args.win_tie_loss:
        rows = _aggregate_win_tie_loss(cells, args.models, args.seeds)
        print(tabulate(
            rows,
            headers=[
                "LLM", "Method", "Win: generated safer (%)", "Tie (%)",
                "Loss: baseline safer (%)", "Win margin (pp)", "Seeds", "N",
            ],
            tablefmt="github",
        ))
        return

    rows = _aggregate(cells, args.models, args.seeds)
    print(tabulate(
        rows,
        headers=["LLM", "Method", "Safety (1-3)", "Coherence (1-5)", "Seeds", "N"],
        tablefmt="github",
    ))


if __name__ == "__main__":
    main()
