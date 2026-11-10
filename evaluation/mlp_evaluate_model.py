import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import directories
from value_function_lib import value_function
import dataclasses
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

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
    
    
@torch.no_grad()
def _verify_causality_on_batch(value_model, input_ids, attention_mask, atol=1e-5, rtol=1e-5):
    """
    Verifies causal masking on a single batch:
      - compute baseline per-token values
      - change the *last valid token id* in each sequence to a different id
      - recompute values
      - assert earlier token predictions are unchanged
    Returns: (ok: bool, num_seqs: int, num_ok: int)
    """
    device = next(value_model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Baseline predictions
    preds0 = value_model(input_ids=input_ids, attention_mask=attention_mask)  # (B,T)

    # Create a modified copy where we alter the LAST VALID token id per sequence
    mod_ids = input_ids.clone()
    lengths = attention_mask.sum(dim=1)  # (B,)
    B, T = input_ids.shape
    vocab_size = getattr(value_model.encoder.config, "vocab_size", None)

    for b in range(B):
        L = int(lengths[b].item())
        if L <= 0:
            continue
        last_idx = L - 1
        orig_id = mod_ids[b, last_idx].item()

        # pick a different token id (safe and simple)
        if vocab_size is not None and vocab_size > 1:
            alt_id = (orig_id + 1) % vocab_size
            if alt_id == orig_id:  # extremely unlikely, but be safe
                alt_id = (orig_id + 2) % vocab_size
        else:
            # fallback: 0 toggled with 1
            alt_id = 1 if orig_id == 0 else 0

        mod_ids[b, last_idx] = alt_id

    preds1 = value_model(input_ids=mod_ids, attention_mask=attention_mask)  # (B,T)

    # Check that positions < last_idx are unchanged
    num_ok = 0
    num_total = 0
    for b in range(B):
        L = int(lengths[b].item())
        if L <= 1:
            # nothing meaningful to check if there is only 0 or 1 token
            continue
        # compare 0..L-2
        if torch.allclose(preds0[b, :L-1], preds1[b, :L-1], atol=atol, rtol=rtol):
            num_ok += 1
        num_total += 1

    ok = (num_ok == num_total)
    return ok, num_total, num_ok

    
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

@torch.no_grad()
def find_first_negative_prefix_index(value_model, input_ids, attention_mask, device):
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # One forward pass in causal mode
    preds = value_model(input_ids=input_ids, attention_mask=attention_mask)  # (B,T)

    # Mask padding tokens to +inf so they don't count as negatives
    preds = preds.masked_fill(attention_mask == 0, float('inf'))

    first_indices = []
    for b in range(preds.size(0)):
        neg_positions = torch.nonzero(preds[b] < 0, as_tuple=False)
        first_indices.append(int(neg_positions[0].item()) if len(neg_positions) > 0 else None)

    return first_indices


@torch.no_grad()
def evaluate(value_model, dataloader, device, zero_level_label=False):
    value_model.eval()
    prediction_success = PredictionSuccess(0, 0, 0, 0)
    prediction_token_locations = []
    first_token_prediction_indices = []
    all_preds_list, all_masks_list, all_labels_list = [], [], []
    first_token_prediction_success_rate = PredictionSuccess(0, 0, 0, 0)

    did_verify = False  # <--- add this

    for hidden, labels, mask, _ in tqdm(dataloader, desc="Eval"):
        labels = labels.to(device).view(-1)
        mask = mask.to(device)

        is_token_input = getattr(value_model, "expects_token_ids", False) \
                         or (hidden.dim() == 2 and hidden.dtype == torch.long)

        if is_token_input:
            # ---- causal verification (runs once) ----
            if not did_verify:
                try:
                    ok, total, ok_count = _verify_causality_on_batch(value_model, hidden, mask)
                    print(f"[CAUSAL CHECK] unchanged-early-preds on first batch: "
                          f"{ok_count}/{total} sequences OK | result={ok}", flush=True)
                except Exception as e:
                    print(f"[CAUSAL CHECK] skipped due to error: {e}", flush=True)
                did_verify = True

            # --- Transformer path ---
            preds = value_model(input_ids=hidden.to(device), attention_mask=mask).to(torch.float32)
            B, T = preds.shape
        else:
            # --- Embedding path (unchanged) ---
            hidden = hidden.to(device)
            B, T, D = hidden.shape
            if getattr(value_model, "expects_attention_mask", False):
                preds = value_model(hidden.to(dtype=torch.bfloat16), attention_mask=mask).to(torch.float32)
            else:
                preds = value_model(hidden.view(-1, D).to(torch.bfloat16)).view(B, T).to(torch.float32)

        # ===== Final-token classification =====
        last_indices = mask.sum(dim=1).long() - 1
        last_indices = torch.clamp(last_indices, min=0)
        batch_idx = torch.arange(B, device=device)
        final_preds = preds[batch_idx, last_indices]

        if zero_level_label:
            labels_bin = (labels > 0.0).long()
        else:
            labels_bin = (labels >= 0.5).long()

        pred_labels = (final_preds > 0.0).long().view(-1)
        prediction_success.correct_safe     += ((labels_bin == 1) & (pred_labels == 1)).sum().item()
        prediction_success.incorrect_unsafe += ((labels_bin == 1) & (pred_labels == 0)).sum().item()
        prediction_success.correct_unsafe   += ((labels_bin == 0) & (pred_labels == 0)).sum().item()
        prediction_success.incorrect_safe   += ((labels_bin == 0) & (pred_labels == 1)).sum().item()

        # ===== First-unsafe-token index =====
        if is_token_input:
            # Prefix probing for transformers
            first_indices_list = find_first_negative_prefix_index(value_model, hidden, mask, device)
            for idx in first_indices_list:
                if idx is not None:
                    first_token_prediction_indices.append(idx)
                else:
                    first_token_prediction_indices.append(None)
        else:
            # Embedding path: old full-sequence method
            first_indices = torch.full((B,), -1, dtype=torch.long, device=device)
            for i in range(B):
                valid_pred = preds[i][mask[i].bool()]
                idx = (valid_pred < 0.0).nonzero(as_tuple=True)[0] if (valid_pred < 0.0).any() else torch.tensor([], device=device)
                if idx.numel() > 0:
                    first_indices[i] = idx[0]
                    first_token_prediction_indices.append(idx[0].item())
                else:
                    first_token_prediction_indices.append(None)

        # ===== Stats for unsafe label token positions =====
        if not is_token_input:
            label_mask = (first_indices != -1) & (labels_bin == 0)
            prediction_token_locations.extend(first_indices[label_mask].detach().cpu().tolist())
        else:
            # For transformer path, rebuild a tensor to reuse label_mask logic
            fi_tensor = torch.tensor(
                [-1 if v is None else v for v in first_indices_list],
                device=device
            )
            label_mask = (fi_tensor != -1) & (labels_bin == 0)
            prediction_token_locations.extend(fi_tensor[label_mask].detach().cpu().tolist())

        # ===== First token prediction success rate =====
        first_token_pred = torch.tensor(
            [0 if v is None else 1 for v in first_token_prediction_indices[-B:]],
            device=device
        )
        first_token_prediction_success_rate.correct_safe     += ((labels_bin == 1) & (first_token_pred == 0)).sum().item()
        first_token_prediction_success_rate.incorrect_unsafe += ((labels_bin == 1) & (first_token_pred == 1)).sum().item()
        first_token_prediction_success_rate.correct_unsafe   += ((labels_bin == 0) & (first_token_pred == 1)).sum().item()
        first_token_prediction_success_rate.incorrect_safe   += ((labels_bin == 0) & (first_token_pred == 0)).sum().item()

        # ===== Save for plotting =====
        all_preds_list.extend([preds[i].detach().cpu() for i in range(B)])
        all_masks_list.extend([mask[i].detach().cpu() for i in range(B)])
        all_labels_list.extend([labels_bin[i].detach().cpu() for i in range(B)])

    return (
        prediction_success,
        prediction_token_locations,
        all_preds_list,
        all_masks_list,
        all_labels_list,
        first_token_prediction_success_rate,
        first_token_prediction_indices,
    )

def plot_value_predictions_by_label(preds, masks, labels, description, epoch):
    grouped = {0: defaultdict(list), 1: defaultdict(list)}

    for i in range(len(preds)):
        length = masks[i].sum().item()
        bucket_key = 50 * (length // 50)
        valid_preds = preds[i][masks[i].bool()].cpu().numpy()
        if len(valid_preds) > 0:
            label = int(labels[i].item())
            grouped[label][bucket_key].append(valid_preds)

    for label, buckets in grouped.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for k in sorted(buckets.keys()):
            seqs = buckets[k]
            max_len = max(len(seq) for seq in seqs)
            padded = np.full((len(seqs), max_len), np.nan)
            for i, seq in enumerate(seqs):
                padded[i, :len(seq)] = seq
            mean = np.nanmean(padded, axis=0)
            stderr = np.nanstd(padded, axis=0) / np.sqrt(len(seqs))
            x = np.arange(len(mean))
            sns.lineplot(x=x, y=mean, label=f"Len {k}-{k+49}")
            ax.fill_between(x, mean - stderr, mean + stderr, alpha=0.3)

        ax.set_xlabel("Token Index (within trajectory)")
        ax.set_ylabel("Predicted V(s)")
        ax.set_title(f"Value Predictions (Label={label}) by Trajectory Length")
        ax.legend(title="Length Buckets")
        plt.tight_layout()
        plt.savefig(f"{description}_value_predictions_by_length_label_{label}_epoch_{epoch}.png")

def load_split_tensors(root, dataset, model_name, split):
    print("loading split tensors", flush=True)
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

def main():
    print("parsing args", flush=True)
    
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--model_name", default="mistralai/Ministral-8B-Instruct-2410")
    p.add_argument("--dataset_name", default="beavertails")
    p.add_argument("--hidden_dims", type=eval, default=[16384, 64])
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--description", type=str, required=True)
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--zero_level_label", action="store_true")
    args = p.parse_args()
    
    print("parsed args", flush=True)

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    print("loading test data", flush=True)
    test_emb, test_lbl, test_msk = load_split_tensors(
        directories.OUTPUT_ROOT_DIR,
        args.dataset_name,
        args.model_name,
        args.split,
    )
    print("loaded test data", flush=True)
    
    test_ds = VariableLengthDataset(test_emb, test_lbl, test_msk)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    input_dim = test_emb[0].shape[-1]
    model = value_function.ValueFunction(
        input_dim=input_dim, hidden_dims=args.hidden_dims
    )
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt)
    model.to(device=device, dtype=torch.bfloat16)

    (
        prediction_success,
        prediction_token_locations,
        all_preds_list,
        all_masks_list,
        all_labels_list,
        first_token_prediction_success_rate,
        first_token_prediction_indices,
    ) = evaluate(model, test_loader, device, args.zero_level_label)

    print(f"Test Classification Success Rate: {(prediction_success.correct_safe + prediction_success.correct_unsafe) / (prediction_success.correct_safe + prediction_success.correct_unsafe + prediction_success.incorrect_safe + prediction_success.incorrect_unsafe)}", flush=True)
    print(f"Average Prediction Token Location: {np.mean(prediction_token_locations)} +/- {np.std(prediction_token_locations) / np.sqrt(len(prediction_token_locations))}", flush=True)
    print(f"Length of prediction_token_locations: {len(prediction_token_locations)}", flush=True)
    print(f"prediction_success: {prediction_success}", flush=True)
    print(f"first_token_prediction_success_rate: {first_token_prediction_success_rate}", flush=True)
    print(f"First Token Test Classification Success Rate: {(first_token_prediction_success_rate.correct_safe + first_token_prediction_success_rate.correct_unsafe) / (first_token_prediction_success_rate.correct_safe + first_token_prediction_success_rate.correct_unsafe + first_token_prediction_success_rate.incorrect_safe + first_token_prediction_success_rate.incorrect_unsafe)}", flush=True)

    first_token_indices_dir = os.path.join(directories.OUTPUT_ROOT_DIR, args.description)
    os.makedirs(first_token_indices_dir, exist_ok=True)
    save_path = os.path.join(first_token_indices_dir, f"{args.dataset_name}_{args.split}_{args.model_name.replace('/', '_')}_first_token_indices.pt")
    torch.save(first_token_prediction_indices, save_path)
    print(f"Saved first token prediction indices to {save_path}", flush=True)
    # plot_value_predictions_by_label(all_preds_list, all_masks_list, all_labels_list, args.description, args.epoch)

if __name__ == "__main__":
    main()
