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
def evaluate(value_model, dataloader, device, zero_level_label=False):
    value_model.eval()
    prediction_success = PredictionSuccess(0, 0, 0, 0)
    prediction_token_locations = []
    first_token_prediction_indices = []
    all_preds_list = []
    all_masks_list = []
    all_labels_list = []
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
            labels = (labels > 0.0).long()
            
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
