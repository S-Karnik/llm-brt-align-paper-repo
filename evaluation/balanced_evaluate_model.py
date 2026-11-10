import argparse
import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import json

import directories
from value_function_lib import value_function
import dataclasses
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from statistics import mean, stdev

def aggregate_value_predictions_by_label(preds_all, masks_all, labels_all):
    """
    Aggregates predicted value trajectories by label (0/1) and trajectory length bucket.
    """
    grouped = {0: defaultdict(list), 1: defaultdict(list)}

    for preds, mask, label in zip(preds_all, masks_all, labels_all):
        length = mask.sum().item()
        bucket_key = 50 * (length // 50)
        valid_preds = preds[mask.bool()].cpu().numpy()
        if len(valid_preds) > 0:
            label_int = int(label.item() > 0.5) if torch.is_tensor(label) else int(label > 0.5)
            grouped[label_int][bucket_key].append(valid_preds)

    return grouped


def plot_aggregated_value_predictions(grouped, model_name, dataset_name, description, epoch):
    for label, buckets in grouped.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for k in sorted(buckets.keys()):
            seqs = buckets[k]
            max_len = max(len(seq) for seq in seqs)
            padded = np.full((len(seqs), max_len), np.nan)
            for i, seq in enumerate(seqs):
                padded[i, :len(seq)] = seq
            print("padded shape:", padded.shape)
            mean = np.nanmean(padded, axis=0)
            stderr = np.nanstd(padded, axis=0) / np.sqrt(len(seqs))
            x = np.arange(len(mean))
            print(f"Bucket {k}-{k+49}: mean.shape = {mean.shape}, mean = {mean}")
            sns.lineplot(x=x, y=mean, label=f"V(s) for Seq Len {k}-{k+49}")
            ax.fill_between(x, mean - stderr, mean + stderr, alpha=0.3)

        ax.set_xlabel("Token Index (within trajectory)")
        ax.set_ylabel("Predicted V(s)")
        ax.set_title(f"Average Value Predictions (Label={label}) by Trajectory Length")
        ax.axhline(0, color='grey', linestyle='--', linewidth=1, label="V(s) = 0 (BRT Threshold)")
        ax.legend(title="Value Predictions")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        os.makedirs("value_function_plots_fp32", exist_ok=True)
        plot_path = f"value_function_plots_fp32/{description}_{model_name}_{dataset_name}_value_predictions_by_length_label_{label}_epoch_{epoch}.png"
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Saved value predictions to {plot_path}")

@dataclasses.dataclass
class PredictionSuccess:
    correct_unsafe: int
    incorrect_unsafe: int
    correct_safe: int
    incorrect_safe: int

class VariableLengthDataset(Dataset):
    def __init__(self, embeddings, labels, masks):
        self.embeddings, self.labels, self.masks = embeddings, labels, masks

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.masks[idx]

def collate_fn(batch):
    emb, lbl, msk = zip(*batch)
    lbl = [torch.tensor(l) if isinstance(l, tuple) else l for l in lbl]
    lbl = [l if isinstance(l, torch.Tensor) else torch.tensor([l]) for l in lbl]
    return (
        pad_sequence(emb, batch_first=True),
        torch.stack(lbl),
        pad_sequence(msk, batch_first=True),
    )

def bootstrap_balanced_subsamples(embeddings, labels, masks, num_samples=10, sample_size=100, seed=42):
    rng = torch.Generator().manual_seed(seed)
    labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
    pos_indices = (labels >= 0.5).nonzero(as_tuple=True)[0]
    neg_indices = (labels < 0.5).nonzero(as_tuple=True)[0]

    all_samples = []
    for _ in range(num_samples):
        pos = pos_indices[torch.randint(0, len(pos_indices), (sample_size,), generator=rng)]
        neg = neg_indices[torch.randint(0, len(neg_indices), (sample_size,), generator=rng)]
        indices = torch.cat([pos, neg])
        indices = indices[torch.randperm(len(indices), generator=rng)]
        boot_emb = [embeddings[i] for i in indices]
        boot_lbl = labels[indices]
        boot_msk = [masks[i] for i in indices]
        all_samples.append((boot_emb, boot_lbl, boot_msk))
    return all_samples

@torch.no_grad()
def evaluate(value_model, dataloader, device, zero_level_label=False):
    prediction_success = PredictionSuccess(0, 0, 0, 0)
    prediction_token_locations = []
    all_preds_list = []
    all_masks_list = []
    all_labels_list = []
    first_token_prediction_indices = []
    first_token_prediction_success_rate = PredictionSuccess(0, 0, 0, 0)

    for hidden, labels, mask in tqdm(dataloader, desc="Eval"):
        hidden = hidden.to(device)
        labels = labels.to(device).view(-1)
        mask = mask.to(device)

        B, T, D = hidden.shape
        if T == 0:
            continue

        preds = value_model(hidden.view(-1, D).to(torch.bfloat16)).view(B, T).to(torch.float32)
        last_indices = mask.sum(dim=1).long() - 1
        last_indices = torch.clamp(last_indices, min=0)
        batch_idx = torch.arange(B, device=device)
        final_preds = preds[batch_idx, last_indices]
        pred_labels = (final_preds > 0.0).long().view(-1)

        if zero_level_label:
            labels = (labels > 0.0).long().view(-1)

        prediction_success.correct_safe     += ((labels >= 0.5) & (pred_labels == 1)).sum().item()
        prediction_success.incorrect_unsafe += ((labels >= 0.5) & (pred_labels == 0)).sum().item()
        prediction_success.correct_unsafe   += ((labels < 0.5) & (pred_labels == 0)).sum().item()
        prediction_success.incorrect_safe   += ((labels < 0.5) & (pred_labels == 1)).sum().item()

        first_indices = torch.full((B,), -1, dtype=torch.long, device=device)
        for i in range(B):
            valid_pred = preds[i][mask[i].bool()]
            idx = (valid_pred < 0.0).nonzero(as_tuple=True)[0] if (valid_pred < 0.0).any() else torch.tensor([], device=device)
            if idx.numel() > 0:
                first_indices[i] = idx[0]
                first_token_prediction_indices.append(idx[0].item())
            else:
                first_token_prediction_indices.append(None)

        label_mask = (first_indices != -1) & (labels < 0.5)
        prediction_token_locations.extend(first_indices[label_mask].cpu().tolist())

        first_token_pred = (first_indices != -1).long()
        first_token_prediction_success_rate.correct_safe     += ((labels >= 0.5) & (first_token_pred == 0)).sum().item()
        first_token_prediction_success_rate.incorrect_unsafe += ((labels >= 0.5) & (first_token_pred == 1)).sum().item()
        first_token_prediction_success_rate.correct_unsafe   += ((labels < 0.5) & (first_token_pred == 1)).sum().item()
        first_token_prediction_success_rate.incorrect_safe   += ((labels < 0.5) & (first_token_pred == 0)).sum().item()

        all_preds_list.extend([preds[i].detach().cpu() for i in range(B)])
        all_masks_list.extend([mask[i].detach().cpu() for i in range(B)])
        all_labels_list.extend([labels[i].detach().cpu() for i in range(B)])

    return (
        prediction_success,
        prediction_token_locations,
        all_preds_list,
        all_masks_list,
        all_labels_list,
        first_token_prediction_success_rate,
        first_token_prediction_indices,
    )

def run_bootstrap_evaluation(model, bootstrap_sets, batch_size, device, zero_level_label):
    results = {
        "success_rates": [],
        "unsafe_success_rates": [],
        "safe_success_rates": [],
        "first_token_rates": [],
        "first_token_unsafe_rates": [],
        "first_token_safe_rates": [],
        "avg_pred_locs": [],
        "first_token_indices": [],
        "all_preds": [],
        "all_masks": [],
        "all_labels": []
    }

    for i, (emb_i, lbl_i, msk_i) in enumerate(bootstrap_sets):
        print(f"Evaluating bootstrap sample {i+1}/{len(bootstrap_sets)}", flush=True)
        test_loader = DataLoader(
            VariableLengthDataset(emb_i, lbl_i, msk_i),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        (
            pred_success,
            pred_locs,
            all_preds,
            all_masks,
            all_labels,
            first_token_success,
            first_token_indices
        ) = evaluate(
            model, test_loader, device, zero_level_label=zero_level_label
        )

        n_total = sum(vars(pred_success).values())
        cls_success = (pred_success.correct_safe + pred_success.correct_unsafe) / n_total if n_total > 0 else 0.0
        results["success_rates"].append(cls_success)

        unsafe_total = pred_success.correct_unsafe + pred_success.incorrect_safe
        unsafe_success = pred_success.correct_unsafe / unsafe_total if unsafe_total > 0 else 0.0
        results["unsafe_success_rates"].append(unsafe_success)

        safe_total = pred_success.correct_safe + pred_success.incorrect_unsafe
        safe_success = pred_success.correct_safe / safe_total if safe_total > 0 else 0.0
        results["safe_success_rates"].append(safe_success)

        n_first = sum(vars(first_token_success).values())
        first_token_success_rate = (first_token_success.correct_safe + first_token_success.correct_unsafe) / n_first if n_first > 0 else 0.0
        results["first_token_rates"].append(first_token_success_rate)

        unsafe_first_total = first_token_success.correct_unsafe + first_token_success.incorrect_safe
        unsafe_first_success = first_token_success.correct_unsafe / unsafe_first_total if unsafe_first_total > 0 else 0.0
        results["first_token_unsafe_rates"].append(unsafe_first_success)

        safe_first_total = first_token_success.correct_safe + first_token_success.incorrect_unsafe
        safe_first_success = first_token_success.correct_safe / safe_first_total if safe_first_total > 0 else 0.0
        results["first_token_safe_rates"].append(safe_first_success)

        results["avg_pred_locs"].append(mean(pred_locs) if pred_locs else 0.0)
        results["first_token_indices"].append(first_token_indices)
        
        results["all_preds"].append(all_preds)
        results["all_masks"].append(all_masks)
        results["all_labels"].append(all_labels)

    print("\n=== Bootstrap Evaluation ===")
    print(f"Classification Success Rate:            {100 * mean(results['success_rates']):.2f}% ± {100 * stdev(results['success_rates']) / (len(bootstrap_sets)**0.5):.2f}%")
    print(f" - Safe Classification Success Rate:    {100 * mean(results['safe_success_rates']):.2f}% ± {100 * stdev(results['safe_success_rates']) / (len(bootstrap_sets)**0.5):.2f}%")
    print(f" - Unsafe Classification Success Rate:  {100 * mean(results['unsafe_success_rates']):.2f}% ± {100 * stdev(results['unsafe_success_rates']) / (len(bootstrap_sets)**0.5):.2f}%")
    print(f"First Token Success Rate:               {100 * mean(results['first_token_rates']):.2f}% ± {100 * stdev(results['first_token_rates']) / (len(bootstrap_sets)**0.5):.2f}%")
    print(f" - Safe First Token Success Rate:       {100 * mean(results['first_token_safe_rates']):.2f}% ± {100 * stdev(results['first_token_safe_rates']) / (len(bootstrap_sets)**0.5):.2f}%")
    print(f" - Unsafe First Token Success Rate:     {100 * mean(results['first_token_unsafe_rates']):.2f}% ± {100 * stdev(results['first_token_unsafe_rates']) / (len(bootstrap_sets)**0.5):.2f}%")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--model_name", default="mistralai/Ministral-8B-Instruct-2410")
    parser.add_argument("--dataset_name", default="beavertails")
    parser.add_argument("--hidden_dims", type=eval, default=[16384, 64])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--description", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_bootstrap_samples", type=int, default=40)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--results_descriptor", type=str, required=True)
    parser.add_argument("--zero_level_label", action="store_true")
    args = parser.parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    def load_split_tensors(root, dataset, model_name, split):
        if "beavertails" in dataset:
            split = "330k_" + split
        prefix = f"{root}/{dataset}/{model_name.replace('/', '_')}/{split}"
        if len(split) == 0:
            embeddings = torch.load(f"{prefix}embeddings.pt")
            masks = torch.load(f"{prefix}masks.pt")
            labels = torch.load(f"{prefix}is_safes.pt")
        else:
            embeddings = torch.load(f"{prefix}_embeddings.pt")
            masks = torch.load(f"{prefix}_masks.pt")
            labels = torch.load(f"{prefix}_is_safes.pt")
        return embeddings, labels, masks

    test_emb, test_lbl, test_msk = load_split_tensors(
        directories.DATA_CACHE_DIR,
        args.dataset_name,
        args.model_name,
        args.split,
    )

    bootstrap_sets = bootstrap_balanced_subsamples(
        test_emb, test_lbl, test_msk,
        num_samples=args.num_bootstrap_samples,
        sample_size=args.sample_size,
    )

    input_dim = test_emb[0].shape[-1]
    model = value_function.ValueFunction(input_dim=input_dim, hidden_dims=args.hidden_dims)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt)
    model.to(device=device, dtype=torch.bfloat16)

    results = run_bootstrap_evaluation(model, bootstrap_sets, args.batch_size, device, args.zero_level_label)

    os.makedirs(args.description, exist_ok=True)
    fpath = os.path.join(args.description, f"{args.dataset_name}_{args.split}_{args.model_name.replace('/', '_')}_first_token_indices.pkl")
    print(f"Saving first token indices to {fpath}")
    value_preds_np = [
        [p.detach().cpu().numpy() for p in row] for row in results["all_preds"]
    ]
    masks_np = [
        [m.detach().cpu().numpy() for m in row] for row in results["all_masks"]
    ]
    results_subset = {
        "value_preds": value_preds_np,
        "masks": masks_np,
        "first_token_indices": results["first_token_indices"],
        "labels": (np.array(results["all_labels"]) >= 0.5).astype(int),
    }

    with open(fpath, "wb") as f:
        pickle.dump(results_subset, f)

    PLOT_DATASET_MAP = {
        "beavertails_llm_generated": "BeaverTails",
        "real_toxicity_llm_generated": "RealToxicity",
        "ultrasafety_llm_generated": "UltraSafety",
    }

    RESULTS_DIR = os.environ.get("LLM_REACHABILITY_RESULTS_DIR", "analysis/results_fp32_embeddings")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, f"{args.model_name.replace('/', '_')}_results.json")

    dataset_key = PLOT_DATASET_MAP.get(args.dataset_name)
    if not dataset_key:
        exit()

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}
        
    if dataset_key not in all_results:
        all_results[dataset_key] = {}

    from itertools import chain

    # Flatten and clean token indices
    flat_token_indices = list(filter(
        lambda x: x is not None,
        chain.from_iterable(results["first_token_indices"])
    ))

    # Precompute token index stats
    include_token_index = len(flat_token_indices) > 0
    if include_token_index:
        mean_token_index = mean(flat_token_indices)
        stderr_token_index = stdev(flat_token_indices) / (len(flat_token_indices) ** 0.5)

    # Now loop over metrics
    for metric_name, values, scale in [
        ("Unsafe Classification Success Rate", results["first_token_unsafe_rates"], 1),
        ("Safe Classification Success Rate", results["first_token_safe_rates"], 1),
    ]:
        mean_val = scale * mean(values)
        stderr_val = scale * stdev(values) / (args.num_bootstrap_samples ** 0.5)

        result_entry = {
            "name": args.results_descriptor,
            "mean": mean_val,
            "stderr": stderr_val,
        }

        if metric_name == "Unsafe Classification Success Rate" and include_token_index:
            result_entry["first_token_index"] = {
                "mean": mean_token_index,
                "stderr": stderr_token_index
            }

        if metric_name not in all_results[dataset_key]:
            all_results[dataset_key][metric_name] = []

        all_results[dataset_key][metric_name] = [
            r for r in all_results[dataset_key][metric_name]
            if r["name"] != result_entry["name"]
        ]
        all_results[dataset_key][metric_name].append(result_entry)
        
        print(f"Recorded {metric_name} for {dataset_key} under '{args.results_descriptor}'")
        print(f"result_entry: {result_entry}")


    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved evaluation result to {json_path}", flush=True)
    
    # Flatten across bootstrap rounds
    flat_preds = [item for sublist in results["all_preds"] for item in sublist]
    flat_masks = [item for sublist in results["all_masks"] for item in sublist]
        
    flat_labels = []
    for sublist in results["all_labels"]:
        for item in sublist:
            if isinstance(item, torch.Tensor):
                if item.dim() == 0:
                    flat_labels.append(item.item())
                elif item.dim() == 1 and item.numel() == 1:
                    flat_labels.append(item[0].item())
                else:
                    raise ValueError(f"Unexpected tensor shape: {item.shape}")
            else:
                flat_labels.append(float(item))  # fallback

    grouped = aggregate_value_predictions_by_label(flat_preds, flat_masks, flat_labels)
    plot_aggregated_value_predictions(grouped, model_name=args.model_name.replace('/', '_'), dataset_name=args.dataset_name, description=args.description, epoch=args.epoch)

