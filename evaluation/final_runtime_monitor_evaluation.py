import argparse
import numpy as np
import os
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import json
import dataclasses
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from statistics import mean, stdev

import directories
from value_function_lib import value_function
try:
    from safety_polytope.polytope import lm_constraints
except ImportError:
    lm_constraints = None
import sys

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
    lbl = [torch.tensor(l) if not isinstance(l, torch.Tensor) else l for l in lbl]
    lbl = [l if l.dim() > 0 else l.unsqueeze(0) for l in lbl]
    lengths = torch.tensor([item.shape[0] for item in emb], dtype=torch.long)
    mask_counts = torch.tensor([item.sum().item() for item in msk], dtype=torch.long)
    if not torch.equal(lengths, mask_counts):
        raise ValueError(
            "Mask token counts do not match response embedding lengths: "
            f"lengths={lengths.tolist()}, mask_counts={mask_counts.tolist()}"
        )
    padded_embeddings = pad_sequence(emb, batch_first=True)
    response_masks = (
        torch.arange(padded_embeddings.shape[1]).unsqueeze(0)
        < lengths.unsqueeze(1)
    )
    return (
        padded_embeddings,
        torch.stack(lbl),
        response_masks,
    )


@torch.no_grad()
def evaluate(value_model, dataloader, device, zero_level_label=False):
    prediction_success = PredictionSuccess(0, 0, 0, 0)
    prediction_token_locations = []
    first_token_prediction_indices = []
    first_token_prediction_success_rate = PredictionSuccess(0, 0, 0, 0)
    all_labels = []

    for hidden, labels, mask in dataloader:
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
        
        for label in labels:
            all_labels.append(label.item())

    return first_token_prediction_success_rate, first_token_prediction_indices, all_labels


@torch.no_grad()
def evaluate_safety_polytope(safety_model, dataloader, device):
    safety_model.eval()
    first_token_success = PredictionSuccess(0, 0, 0, 0)
    final_token_success = PredictionSuccess(0, 0, 0, 0)
    prediction_token_locations = []
    first_token_prediction_indices = []
    all_labels = []

    for embeddings, labels, masks in dataloader:
        # Flatten all valid token embeddings across batch
        token_embeddings = []
        final_token_embeddings = []
        mapping = []  # (example_idx, token_idx)
        for i, (emb, mask) in enumerate(zip(embeddings, masks)):
            valid_len = mask.sum().item()
            for t in range(valid_len):
                token_embeddings.append(emb[t])
                mapping.append((i, t))
            final_token_embeddings.append(emb[valid_len - 1])

        if len(token_embeddings) == 0:
            continue

        token_embeddings_tensor = torch.stack(token_embeddings).to(device, dtype=torch.bfloat16)
        safety_preds, cost = safety_model.get_safety_prediction(token_embeddings_tensor, return_cost=True)
        safety_preds = safety_preds.bool().cpu()
        final_safety_preds, _ = safety_model.get_safety_prediction(
            torch.stack(final_token_embeddings).to(device, dtype=torch.bfloat16),
            return_cost=True
        )
        final_safety_preds = final_safety_preds.bool().cpu()
                
        # Reconstruct is_safe lists for each example
        unsafe_token_indices_by_example = defaultdict(list)
        for (ex_idx, tok_idx), is_safe in zip(mapping, safety_preds):
            if not is_safe:
                unsafe_token_indices_by_example[ex_idx].append(tok_idx)

        for i, label in enumerate(labels):
            if label >= 0.5:
                if i not in unsafe_token_indices_by_example:
                    first_token_success.correct_safe += 1
                    first_token_prediction_indices.append(None)
                else:
                    first_token_success.incorrect_unsafe += 1
                    first_token_prediction_indices.append(unsafe_token_indices_by_example[i][0])
                
                if final_safety_preds[i]:
                    final_token_success.correct_safe += 1
                else:
                    final_token_success.incorrect_unsafe += 1
            else:
                if i in unsafe_token_indices_by_example:
                    first_token_success.correct_unsafe += 1
                    first_unsafe = unsafe_token_indices_by_example[i][0]
                    prediction_token_locations.append(first_unsafe)
                    first_token_prediction_indices.append(first_unsafe)
                else:
                    first_token_success.incorrect_safe += 1
                    first_token_prediction_indices.append(None)
                
                if not final_safety_preds[i]:
                    final_token_success.correct_unsafe += 1
                else:
                    final_token_success.incorrect_safe += 1

            all_labels.append(label.item())
    
    return first_token_success, first_token_prediction_indices, final_token_success, all_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--model_name", default="mistralai/Ministral-8B-Instruct-2410")
    parser.add_argument("--dataset_name", default="beavertails")
    parser.add_argument("--hidden_dims", type=eval, default=[16384, 64])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--zero_level_label", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method_type", type=str, required=True)
    parser.add_argument(
        "--embedding_layer_idx",
        type=int,
        default=None,
        help=(
            "Load embeddings from dataset directories suffixed with this layer, "
            "such as beavertails_llm_generated_layer_10."
        ),
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help="Override the directory where runtime-monitor result JSONs are saved.",
    )
    parser.add_argument("--use_final_layer", action="store_true",
                        help="If set, uses the final layer representation instead of the 20th layer (as is the default choice).")

    args = parser.parse_args()

    parsed_model_name = args.model_name.replace('/', '_')
    
    if args.use_final_layer and args.embedding_layer_idx is not None:
        parser.error("--use_final_layer and --embedding_layer_idx cannot be used together")

    if args.results_root is not None:
        save_dir = os.path.join(args.results_root, f"seed_{args.seed}")
    elif args.use_final_layer:
        save_dir = os.path.join(
            directories.DATA_CACHE_DIR,
            "final_runtime_monitor_results_lastlayer",
            f"seed_{args.seed}",
        )
    else:
        save_dir = os.path.join(
            directories.DATA_CACHE_DIR,
            "final_runtime_monitor_results",
            f"seed_{args.seed}",
        )
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, f"{parsed_model_name}.json")
    results_data = dict()

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            loaded_json = json.load(f)

        assert isinstance(loaded_json, dict)
        
        if (args.method_type in loaded_json) and (args.dataset_name in loaded_json[args.method_type]) and ("total_classification_rate" in loaded_json[args.method_type][args.dataset_name]):
            print(f"Already computed runtime monitor results.")
            # sys.exit(0)
        
        results_data = loaded_json

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    def load_split_tensors(root, dataset, model_name, split):
        if "beavertails" in dataset:
            split = "330k_" + split
        dataset_dir = f"{dataset}_llm_generated"
        if args.embedding_layer_idx is not None:
            dataset_dir += f"_layer_{args.embedding_layer_idx}"
        prefix = f"{root}/{dataset_dir}/{model_name.replace('/', '_')}/{split}"
        if args.use_final_layer:
            embeddings = torch.load(f"{prefix}_embeddings_lastlayer.pt")
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
    test_loader = DataLoader(
        VariableLengthDataset(test_emb, test_lbl, test_msk),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    input_dim = test_emb[0].shape[-1]
    final_pred_success = None

    if args.method_type.lower() == "sap":
        if lm_constraints is None:
            raise ImportError(
                "SaP runtime-monitor evaluation requires the external "
                "safety_polytope package."
            )
        model_result = torch.load(args.checkpoint_path, map_location=device, weights_only=False)

        # Get input dim from test embedding
        rep_dim = test_emb[0].shape[-1]

        # Get output dim from saved feature extractor
        if model_result.feature_extractor is not None:
            feature_layers = list(model_result.feature_extractor.children())
            if isinstance(feature_layers[0], torch.nn.Linear):
                feature_dim = feature_layers[0].out_features
                use_nonlinear = True
            else:
                feature_dim = rep_dim
                use_nonlinear = False
        else:
            raise ValueError("Checkpoint does not contain a valid feature_extractor")

        # Reconstruct the model with matching dims
        safety_model = lm_constraints.PolytopeConstraint(
            model=None,
            tokenizer=None,
            num_phi=model_result.phi.shape[0],
            train_on_hs=True,
            feature_dim=feature_dim,
            use_nonlinear=use_nonlinear,
        )
        safety_model.rand_init_phi_theta(model_result.phi.shape[0], x=torch.zeros(1, rep_dim).to(device))

        # Load feature extractor weights
        safety_model.feature_extractor.load_state_dict(model_result.feature_extractor.state_dict())
        safety_model.feature_extractor.to(device, dtype=torch.bfloat16)

        # Load remaining weights
        safety_model.phi = model_result.phi
        safety_model.threshold = model_result.threshold
        safety_model = safety_model.to(device, dtype=torch.bfloat16)

        pred_success, all_first_token_indices, final_pred_success, all_labels = evaluate_safety_polytope(safety_model, test_loader, device)
    else:
        model = value_function.ValueFunction(input_dim=input_dim, hidden_dims=args.hidden_dims)
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt)
        model.to(device=device, dtype=torch.bfloat16)

        pred_success, all_first_token_indices, all_labels = evaluate(
            model, test_loader, device, zero_level_label=args.zero_level_label
        )

    save_path = os.path.join(
        save_dir,
        f"{parsed_model_name}_{args.dataset_name}_{args.method_type}_first_token_indices.pkl"
    )
    print(f"Saving first token indices to {save_path}")
    results_subset = {
        "first_token_indices": all_first_token_indices,
        "labels": all_labels,
    }
    with open(save_path, "wb") as f:
        pickle.dump(results_subset, f)
    
    n_total = sum(vars(pred_success).values())
    cls_success = (pred_success.correct_safe + pred_success.correct_unsafe) / n_total if n_total > 0 else 0.0
    unsafe_total = pred_success.correct_unsafe + pred_success.incorrect_safe
    safe_total = pred_success.correct_safe + pred_success.incorrect_unsafe

    print("\n=== Evaluation Results ===")
    print(f"Classification Success Rate:    {100 * cls_success:.2f}%")
    print(f" - Safe Success Rate:           {100 * (pred_success.correct_safe / safe_total if safe_total else 0):.4f}%")
    print(f" - Unsafe Success Rate:         {100 * (pred_success.correct_unsafe / unsafe_total if unsafe_total else 0):.4f}%")

    if final_pred_success is not None:
        n_total_final = sum(vars(final_pred_success).values())
        cls_success_final = (final_pred_success.correct_safe + final_pred_success.correct_unsafe) / n_total_final if n_total_final > 0 else 0.0
        print(f"\nFinal Token Classification Success Rate:    {100 * cls_success_final:.2f}%")
        print(f" - Final Safe Success Rate:           {100 * (final_pred_success.correct_safe / safe_total if safe_total else 0):.4f}%")
        print(f" - Final Unsafe Success Rate:         {100 * (final_pred_success.correct_unsafe / unsafe_total if unsafe_total else 0):.4f}%")

    # F1 score for unsafe detection (treat "unsafe" as positive class)
    tp = pred_success.correct_unsafe
    fn = pred_success.incorrect_safe
    fp = pred_success.incorrect_unsafe
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f" - Precision (unsafe):          {precision:.4f}")
    print(f" - Recall (unsafe):             {recall:.4f}")
    print(f" - F1 Score (unsafe):           {f1_score:.4f}")

    # Save results
    if args.method_type not in results_data:
        results_data[args.method_type] = {}
    if args.dataset_name not in results_data[args.method_type]:
        results_data[args.method_type][args.dataset_name] = {}

    results_data[args.method_type][args.dataset_name]["total_classification_rate"] = cls_success
    results_data[args.method_type][args.dataset_name]["total_safe_classification_rate"] = pred_success.correct_safe / safe_total if safe_total else 0.0
    results_data[args.method_type][args.dataset_name]["total_unsafe_classification_rate"] = pred_success.correct_unsafe / unsafe_total if unsafe_total else 0.0
    results_data[args.method_type][args.dataset_name]["precision_unsafe"] = precision
    results_data[args.method_type][args.dataset_name]["recall_unsafe"] = recall
    results_data[args.method_type][args.dataset_name]["f1_unsafe"] = f1_score

    # Compute statistics for final predictions if available
    if final_pred_success is not None:
        tp_final = final_pred_success.correct_unsafe
        fn_final = final_pred_success.incorrect_safe
        fp_final = final_pred_success.incorrect_unsafe
        precision_final = tp_final / (tp_final + fp_final) if (tp_final + fp_final) > 0 else 0.0
        recall_final = tp_final / (tp_final + fn_final) if (tp_final + fn_final) > 0 else 0.0
        f1_score_final = (2 * precision_final * recall_final / (precision_final + recall_final)) if (precision_final + recall_final) > 0 else 0.0

        print(f"\n=== Final Token Evaluation Results ===")
        print(f"Final Token - Classification Success Rate:    {100 * cls_success_final:.2f}%")
        print(f" - Final Token - Safe Success Rate:           {100 * (final_pred_success.correct_safe / safe_total if safe_total else 0):.4f}%")
        print(f" - Final Token - Unsafe Success Rate:         {100 * (final_pred_success.correct_unsafe / unsafe_total if unsafe_total else 0):.4f}%")
        print(f" - Final Token - Precision (unsafe):          {precision_final:.4f}")
        print(f" - Final Token - Recall (unsafe):             {recall_final:.4f}")
        print(f" - Final Token - F1 Score (unsafe):           {f1_score_final:.4f}")

        results_data[args.method_type][args.dataset_name]["final_total_classification_rate"] = cls_success_final
        results_data[args.method_type][args.dataset_name]["final_total_safe_classification_rate"] = final_pred_success.correct_safe / safe_total if safe_total else 0.0
        results_data[args.method_type][args.dataset_name]["final_total_unsafe_classification_rate"] = final_pred_success.correct_unsafe / unsafe_total if unsafe_total else 0.0
        results_data[args.method_type][args.dataset_name]["final_precision_unsafe"] = precision_final
        results_data[args.method_type][args.dataset_name]["final_recall_unsafe"] = recall_final
        results_data[args.method_type][args.dataset_name]["final_f1_unsafe"] = f1_score_final

    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)    
