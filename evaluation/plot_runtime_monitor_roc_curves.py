#!/usr/bin/env python3
"""Plot mean response-level ROC curves from saved runtime-monitor sweeps."""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import directories

DEFAULT_RESULTS_ROOT = os.path.join(directories.DATA_CACHE_DIR, "final_runtime_monitor_auc_results")
DEFAULT_METHODS = [
    "BRT",
    "Sample-BRT",
    "Re-Control",
    "SaP",
    "Prefix Classifier",
    "Generate-Then-Classify",
]
METHOD_COLORS = {
    "BRT": "#d46322",
    "Sample-BRT": "#ff9500",
    "Re-Control": "#477998",
    "SaP": "#806363",
    "Prefix Classifier": "#2a9d8f",
    "Generate-Then-Classify": "#6a4c93",
}


def normalized_method_name(method):
    return method.lower().replace("-", "_").replace(" ", "_")


def curve_path_for(seed_dir, model_name, dataset_name, method, metrics):
    recorded_path = metrics.get("curve_path")
    if recorded_path and os.path.exists(recorded_path):
        return recorded_path

    filename = (
        f"{model_name.replace('/', '_')}_{dataset_name}_"
        f"{normalized_method_name(method)}_curves.npz"
    )
    fallback = os.path.join(seed_dir, filename)
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(f"Missing saved curve file: {fallback}")


def load_curves(results_root, model_name, dataset_name, methods):
    model_file = model_name.replace("/", "_") + ".json"
    json_paths = sorted(
        glob.glob(os.path.join(results_root, "seed_*", model_file))
    )
    if not json_paths:
        raise FileNotFoundError(
            f"No AUC results named {model_file} under {results_root}"
        )

    curves = {method: [] for method in methods}
    for json_path in json_paths:
        with open(json_path) as input_file:
            results = json.load(input_file)
        for method in methods:
            metrics = results.get(method, {}).get(dataset_name)
            if metrics is None:
                continue
            curve_path = curve_path_for(
                os.path.dirname(json_path),
                model_name,
                dataset_name,
                method,
                metrics,
            )
            with np.load(curve_path) as curve_data:
                curves[method].append(
                    {
                        "fpr": curve_data["fpr"].copy(),
                        "tpr": curve_data["tpr"].copy(),
                        "auroc": float(metrics["auroc"]),
                    }
                )
    return curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--model_name", required=True)
    parser.add_argument(
        "--dataset_name",
        required=True,
        choices=["beavertails", "real_toxicity", "ultrasafety"],
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--num_grid_points", type=int, default=1001)
    parser.add_argument("--no_std_band", action="store_true")
    parser.add_argument("--title", default=None)
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()

    method_curves = load_curves(
        args.results_root,
        args.model_name,
        args.dataset_name,
        args.methods,
    )
    fpr_grid = np.linspace(0.0, 1.0, args.num_grid_points)

    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    plotted = 0
    for method in args.methods:
        seed_curves = method_curves[method]
        if not seed_curves:
            print(f"Skipping {method}: no curves found")
            continue

        interpolated_tprs = np.stack(
            [
                np.interp(fpr_grid, curve["fpr"], curve["tpr"])
                for curve in seed_curves
            ]
        )
        interpolated_tprs[:, 0] = 0.0
        interpolated_tprs[:, -1] = 1.0
        mean_tpr = interpolated_tprs.mean(axis=0)
        std_tpr = interpolated_tprs.std(axis=0, ddof=1) if len(seed_curves) > 1 else np.zeros_like(mean_tpr)
        aurocs = np.asarray([curve["auroc"] for curve in seed_curves])
        auc_std = aurocs.std(ddof=1) if aurocs.size > 1 else 0.0
        color = METHOD_COLORS.get(method)

        axis.plot(
            fpr_grid,
            mean_tpr,
            linewidth=2.2,
            color=color,
            label=(
                f"{method} (AUROC {aurocs.mean():.3f} +/- "
                f"{auc_std:.3f}, n={len(seed_curves)})"
            ),
        )
        if not args.no_std_band and len(seed_curves) > 1:
            axis.fill_between(
                fpr_grid,
                np.maximum(mean_tpr - std_tpr, 0.0),
                np.minimum(mean_tpr + std_tpr, 1.0),
                color=color,
                alpha=0.14,
                linewidth=0,
            )
        plotted += 1

    if plotted == 0:
        raise RuntimeError("No ROC curves were available for the requested methods.")

    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1.2)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.01)
    axis.set_xlabel("False positive rate (safe responses flagged unsafe)")
    axis.set_ylabel("True positive rate (unsafe responses detected)")
    if args.title:
        axis.set_title(args.title)
    else:
        axis.set_title(
            f"Runtime-monitor ROC: {args.model_name} / {args.dataset_name}"
        )
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right", frameon=False)
    figure.tight_layout()

    save_path = args.save_path
    if save_path is None:
        save_path = (
            f"runtime_monitor_roc_{args.model_name.replace('/', '_')}_"
            f"{args.dataset_name}.png"
        )
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved ROC plot to {save_path}")


if __name__ == "__main__":
    main()
