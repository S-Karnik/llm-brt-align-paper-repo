#!/usr/bin/env python3
"""Evaluate response-level runtime monitors across every decision threshold.

The positive class is unsafe. Each response is reduced to one continuous score
that matches the runtime decision rule (flag a response if any token is unsafe):

* Value functions: ``unsafe_score = -min_t V(h_t)``.
* Safety Polytope: ``unsafe_score = max_{t,j}(cost_j(h_t) - threshold_j)``.

Thus, a larger score always means "more unsafe" for every method. Results are
written separately from the fixed-threshold runtime-monitor evaluation.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import directories
try:
    from safety_polytope.polytope import lm_constraints
except ImportError:
    lm_constraints = None
from value_function_lib import value_function


DEFAULT_RESULTS_ROOT = os.path.join(directories.DATA_CACHE_DIR, "final_runtime_monitor_auc_results_float32")


class VariableLengthDataset(Dataset):
    def __init__(self, embeddings, labels, masks):
        if not (len(embeddings) == len(labels) == len(masks)):
            raise ValueError(
                "Embedding, label, and mask lengths differ: "
                f"{len(embeddings)}, {len(labels)}, {len(masks)}"
            )
        self.embeddings = embeddings
        self.labels = labels
        self.masks = masks

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.masks[idx]


def collate_fn(batch):
    embeddings, labels, masks = zip(*batch)
    labels = [torch.as_tensor(label).reshape(-1)[0] for label in labels]
    return (
        pad_sequence(embeddings, batch_first=True),
        torch.stack(labels),
        pad_sequence(masks, batch_first=True),
    )


def labels_to_unsafe(labels, zero_level_label=False):
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    safe = labels > 0.0 if zero_level_label else labels >= 0.5
    return (~safe).astype(np.int64)


@torch.no_grad()
def score_value_model(model, dataloader, device, inference_dtype):
    model.eval()
    unsafe_scores = []
    token_unsafe_scores = []
    labels = []

    for hidden, batch_labels, masks in tqdm(dataloader, desc="Scoring value model"):
        hidden = hidden.to(device=device, dtype=inference_dtype)
        masks = masks.to(device).bool()
        batch_labels = batch_labels.reshape(-1)

        batch_size, num_tokens, hidden_dim = hidden.shape
        predictions = model(hidden.reshape(-1, hidden_dim)).reshape(
            batch_size, num_tokens
        ).float()

        predictions = predictions.masked_fill(~masks, float("inf"))
        minimum_values = predictions.min(dim=1).values
        valid = masks.any(dim=1)

        unsafe_scores.extend((-minimum_values[valid]).cpu().tolist())
        for example_idx in torch.nonzero(valid, as_tuple=False).reshape(-1).tolist():
            token_unsafe_scores.append(
                (-predictions[example_idx][masks[example_idx]]).cpu().numpy()
            )
        labels.extend(batch_labels[valid.cpu()].cpu().tolist())

    return np.asarray(unsafe_scores), np.asarray(labels), token_unsafe_scores


@torch.no_grad()
def score_safety_polytope(model, dataloader, device, inference_dtype):
    model.eval()
    unsafe_scores = []
    token_unsafe_scores = []
    labels = []

    for embeddings, batch_labels, masks in tqdm(
        dataloader, desc="Scoring Safety Polytope"
    ):
        token_embeddings = []
        owners = []
        valid_examples = []

        for example_idx, (embedding, mask) in enumerate(zip(embeddings, masks)):
            valid_tokens = embedding[mask.bool()]
            if valid_tokens.shape[0] == 0:
                continue
            valid_examples.append(example_idx)
            token_embeddings.append(valid_tokens)
            owners.extend([example_idx] * valid_tokens.shape[0])

        if not token_embeddings:
            continue

        flat_embeddings = torch.cat(token_embeddings, dim=0).to(
            device, dtype=inference_dtype
        )
        _, costs = model.get_safety_prediction(flat_embeddings, return_cost=True)
        violations = costs - model.threshold.to(device=device, dtype=costs.dtype)
        token_scores = violations.max(dim=1).values.float().cpu()

        owners = torch.tensor(owners, dtype=torch.long)
        for example_idx in valid_examples:
            example_scores = token_scores[owners == example_idx]
            unsafe_scores.append(example_scores.max().item())
            token_unsafe_scores.append(example_scores.numpy())
            labels.append(torch.as_tensor(batch_labels[example_idx]).item())

    return np.asarray(unsafe_scores), np.asarray(labels), token_unsafe_scores


def save_ragged_token_scores(path, token_scores, raw_labels, inference_dtype):
    lengths = np.asarray([len(scores) for scores in token_scores], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    flat_scores = (
        np.concatenate(token_scores).astype(np.float32, copy=False)
        if token_scores
        else np.asarray([], dtype=np.float32)
    )
    np.savez_compressed(
        path,
        token_unsafe_scores=flat_scores,
        offsets=offsets,
        raw_labels=np.asarray(raw_labels),
        inference_dtype=np.asarray(str(inference_dtype).replace("torch.", "")),
    )


def compute_curves(raw_labels, unsafe_scores, zero_level_label=False):
    unsafe_labels = labels_to_unsafe(raw_labels, zero_level_label)
    if unsafe_scores.shape != unsafe_labels.shape:
        raise ValueError(
            f"Score and label shapes differ: {unsafe_scores.shape}, "
            f"{unsafe_labels.shape}"
        )
    if not np.isfinite(unsafe_scores).all():
        raise ValueError("Non-finite unsafe scores encountered.")
    if np.unique(unsafe_labels).size != 2:
        raise ValueError("AUROC/AUPRC require both safe and unsafe examples.")

    fpr, tpr, roc_thresholds = roc_curve(unsafe_labels, unsafe_scores)
    precision, recall, pr_thresholds = precision_recall_curve(
        unsafe_labels, unsafe_scores
    )

    metrics = {
        "auroc": float(roc_auc_score(unsafe_labels, unsafe_scores)),
        "auprc": float(auc(recall, precision)),
        "average_precision": float(
            average_precision_score(unsafe_labels, unsafe_scores)
        ),
        "num_examples": int(unsafe_labels.size),
        "num_unsafe": int(unsafe_labels.sum()),
        "num_safe": int(unsafe_labels.size - unsafe_labels.sum()),
        "unsafe_prevalence": float(unsafe_labels.mean()),
        "score_orientation": "higher_is_more_unsafe",
        "response_reduction": "any_token_unsafe",
    }
    curves = {
        "unsafe_labels": unsafe_labels,
        "unsafe_scores": unsafe_scores,
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thresholds,
    }
    return metrics, curves


def load_safety_polytope(
    checkpoint_path, representation_dim, device, inference_dtype
):
    if lm_constraints is None:
        raise ImportError(
            "SaP runtime-monitor evaluation requires the external "
            "safety_polytope package."
        )
    saved_model = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if saved_model.feature_extractor is None:
        raise ValueError("SaP checkpoint has no feature extractor.")

    feature_layers = list(saved_model.feature_extractor.children())
    if feature_layers and isinstance(feature_layers[0], torch.nn.Linear):
        feature_dim = feature_layers[0].out_features
        use_nonlinear = True
    else:
        feature_dim = representation_dim
        use_nonlinear = False

    model = lm_constraints.PolytopeConstraint(
        model=None,
        tokenizer=None,
        num_phi=saved_model.phi.shape[0],
        train_on_hs=True,
        feature_dim=feature_dim,
        use_nonlinear=use_nonlinear,
    )
    model.rand_init_phi_theta(
        saved_model.phi.shape[0],
        x=torch.zeros(1, representation_dim, device=device),
    )
    model.feature_extractor.load_state_dict(
        saved_model.feature_extractor.state_dict()
    )
    model.feature_extractor.to(device=device, dtype=inference_dtype)
    model.phi = torch.nn.Parameter(
        saved_model.phi.detach().to(device=device, dtype=inference_dtype),
        requires_grad=False,
    )
    model.threshold = torch.nn.Parameter(
        saved_model.threshold.detach().to(
            device=device, dtype=inference_dtype
        ),
        requires_grad=False,
    )
    return model.to(device=device, dtype=inference_dtype)


def load_split_tensors(root, dataset, model_name, split, use_final_layer=False, embedding_layer_idx=None):
    if "beavertails" in dataset:
        split = "330k_" + split
    dataset_dir = f"{dataset}_llm_generated"
    if embedding_layer_idx is not None:
        dataset_dir += f"_layer_{embedding_layer_idx}"
    prefix = os.path.join(
        root,
        dataset_dir,
        model_name.replace("/", "_"),
        split,
    )
    embedding_suffix = "_embeddings_lastlayer.pt" if use_final_layer else "_embeddings.pt"
    embeddings = torch.load(prefix + embedding_suffix)
    masks = torch.load(prefix + "_masks.pt")
    labels = torch.load(prefix + "_is_safes.pt")
    return embeddings, labels, masks


def result_stem(model_name, dataset_name, method_type):
    method = method_type.lower().replace("-", "_").replace(" ", "_")
    return f"{model_name.replace('/', '_')}_{dataset_name}_{method}"


def atomic_json_dump(data, path):
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary_path.open("w") as output_file:
        json.dump(data, output_file, indent=2, sort_keys=True)
    os.replace(temporary_path, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument(
        "--model_name", default="mistralai/Ministral-8B-Instruct-2410"
    )
    parser.add_argument("--dataset_name", default="beavertails")
    parser.add_argument("--hidden_dims", type=eval, default=[16384, 64])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--split", default="test")
    parser.add_argument("--zero_level_label", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method_type", required=True)
    parser.add_argument(
        "--inference_dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Precision used to generate continuous monitor scores.",
    )
    parser.add_argument("--use_final_layer", action="store_true")
    parser.add_argument(
        "--embedding_layer_idx",
        type=int,
        default=None,
        help=(
            "Load embeddings from dataset directories suffixed with this layer, "
            "such as beavertails_llm_generated_layer_20."
        ),
    )
    parser.add_argument(
        "--results_root", default=DEFAULT_RESULTS_ROOT
    )
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument(
        "--save_token_scores",
        action="store_true",
        help="Save ragged per-token unsafe scores for matched-FAR analyses.",
    )
    args = parser.parse_args()
    if args.use_final_layer and args.embedding_layer_idx is not None:
        parser.error("--use_final_layer and --embedding_layer_idx cannot be used together")

    save_dir = Path(args.results_root) / f"seed_{args.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    parsed_model_name = args.model_name.replace("/", "_")
    json_path = save_dir / f"{parsed_model_name}.json"
    curve_path = save_dir / (
        result_stem(args.model_name, args.dataset_name, args.method_type)
        + "_curves.npz"
    )
    token_score_path = save_dir / (
        result_stem(args.model_name, args.dataset_name, args.method_type)
        + "_token_scores.npz"
    )

    required_outputs_exist = (
        json_path.exists()
        and curve_path.exists()
        and (not args.save_token_scores or token_score_path.exists())
    )
    if required_outputs_exist and not args.force_recompute:
        with json_path.open() as input_file:
            saved_results = json.load(input_file)
        if args.method_type in saved_results and args.dataset_name in saved_results[args.method_type]:
            print(f"Already computed AUC results: {curve_path}")
            return

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    inference_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.inference_dtype]
    embeddings, labels, masks = load_split_tensors(
        directories.DATA_CACHE_DIR,
        args.dataset_name,
        args.model_name,
        args.split,
        args.use_final_layer,
        args.embedding_layer_idx,
    )
    dataloader = DataLoader(
        VariableLengthDataset(embeddings, labels, masks),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    representation_dim = embeddings[0].shape[-1]

    if args.method_type.lower() == "sap":
        model = load_safety_polytope(
            args.checkpoint_path, representation_dim, device, inference_dtype
        )
        unsafe_scores, raw_labels, token_unsafe_scores = score_safety_polytope(
            model, dataloader, device, inference_dtype
        )
        score_definition = "max_token_and_edge_constraint_violation"
    else:
        model = value_function.ValueFunction(
            input_dim=representation_dim, hidden_dims=args.hidden_dims
        )
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        model.to(device=device, dtype=inference_dtype)
        unsafe_scores, raw_labels, token_unsafe_scores = score_value_model(
            model, dataloader, device, inference_dtype
        )
        score_definition = "negative_minimum_token_value"

    metrics, curves = compute_curves(
        raw_labels, unsafe_scores, args.zero_level_label
    )
    metrics["score_definition"] = score_definition
    metrics["inference_dtype"] = args.inference_dtype
    metrics["checkpoint_path"] = args.checkpoint_path
    metrics["curve_path"] = str(curve_path)
    if args.save_token_scores:
        save_ragged_token_scores(
            token_score_path, token_unsafe_scores, raw_labels,
            args.inference_dtype,
        )
        metrics["token_score_path"] = str(token_score_path)

    np.savez_compressed(curve_path, **curves)

    saved_results = {}
    if json_path.exists():
        with json_path.open() as input_file:
            saved_results = json.load(input_file)
    saved_results.setdefault(args.method_type, {})[args.dataset_name] = metrics
    atomic_json_dump(saved_results, json_path)

    print("\n=== Runtime Monitor Threshold-Sweep Results ===")
    print(f"Method:                 {args.method_type}")
    print(f"Dataset:                {args.dataset_name}")
    print(f"Examples:               {metrics['num_examples']}")
    print(f"Unsafe prevalence:      {metrics['unsafe_prevalence']:.6f}")
    print(f"AUROC:                  {metrics['auroc']:.6f}")
    print(f"AUPRC (trapezoidal):    {metrics['auprc']:.6f}")
    print(f"Average precision:      {metrics['average_precision']:.6f}")
    print(f"Saved curves:           {curve_path}")
    if args.save_token_scores:
        print(f"Saved token scores:     {token_score_path}")
    print(f"Saved metrics:          {json_path}")


if __name__ == "__main__":
    main()
