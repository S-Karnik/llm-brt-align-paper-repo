import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader, Dataset
import wandb
import numpy as np
import os
import random
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR

from toxicity_value_function import data_loading_helpers

import directories
from value_function_lib import value_function

from evaluation import mlp_evaluate_model

os.environ['WANDB_START_METHOD'] = 'thread'
wandb.init(project="llm-control", name="train_fitted_value_iteration")

class GammaScheduler:
    def __init__(self, gamma_start, gamma_end, anneal_epochs):
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.anneal_epochs = anneal_epochs

    def get_gamma(self, epoch):
        if epoch >= self.anneal_epochs:
            return self.gamma_end
        pct = epoch / self.anneal_epochs  # ranges from 0 to 1
        return self.gamma_start + (self.gamma_end - self.gamma_start) * pct

class VariableLengthDataset(Dataset):
    def __init__(self, embeddings, labels, masks, scores):
        self.embeddings = embeddings
        self.labels = labels
        self.masks = masks
        self.scores = scores
        lengths = {
            "embeddings": len(self.embeddings),
            "labels": len(self.labels),
            "masks": len(self.masks),
            "scores": len(self.scores),
        }
        print(f"lengths: {lengths}", flush=True)
        if len(set(lengths.values())) != 1:
            raise ValueError(
                "[VariableLengthDataset] length mismatch after loading tensors: "
                f"{lengths}"
            )

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.masks[idx], self.scores[idx]

def collate_fn(batch):
    """
    Each item: (emb_i: [Ti, D], label_i: scalar, mask_i: full-sequence mask,
    scores_i: [Si]).

    The saved embeddings are already response-only, while the saved mask is a
    full prompt+response mask whose sum equals the response length. Build a fresh
    response-local mask here so token values and prefix scores stay aligned.
    """
    aligned_embs = []
    aligned_masks = []
    aligned_scores = []
    labels = []

    for emb, label, mask, scores in batch:
        # convert to tensors if needed
        if not torch.is_tensor(mask):   mask = torch.tensor(mask)
        if not torch.is_tensor(scores): scores = torch.tensor(scores)
        
        # shift scores to [-0.5, 0.5]
        scores = scores - 0.5

        Ti = emb.shape[0]
        Mi = int(mask.sum().item())           # response length implied by mask
        Si = int(scores.shape[0])

        if Mi != Ti:
            raise ValueError(
                f"Mask token count ({Mi}) does not match response embedding length ({Ti})."
            )

        # Effective length: use the smallest available, never invent scores.
        # Prefer the trailing embeddings so existing datasets that included the
        # inserted "\n" as the first response-mask token still align score[t]
        # with the actual response token rather than the separator.
        Teff = min(Ti, Si)

        if Teff == 0:
            # handle degenerate case by creating a single-step dummy to keep loader stable
            # (model should learn nothing from these)
            aligned_embs.append(emb[:1])
            aligned_masks.append(torch.zeros((1,), dtype=mask.dtype, device=mask.device))
            aligned_scores.append(scores[:1] if Si > 0 else scores.new_zeros((1,)))
        else:
            aligned_embs.append(emb[-Teff:])
            aligned_masks.append(torch.ones((Teff,), dtype=mask.dtype, device=mask.device))
            aligned_scores.append(scores[:Teff])

        if not torch.is_tensor(label):
            label = torch.tensor(label)
        labels.append(label.reshape(()))

    # pad to batch
    padded_embeddings = pad_sequence(aligned_embs, batch_first=True).to(torch.bfloat16)
    padded_masks      = pad_sequence(aligned_masks, batch_first=True).to(torch.float32)
    padded_scores     = pad_sequence(aligned_scores, batch_first=True).to(torch.float32)
    labels = torch.stack(labels).to(torch.float32) - 0.5

    return padded_embeddings, labels, padded_masks, padded_scores


def _run_epoch(
    value_model, target_value_model, epoch, warmup_epochs, dataloader, device,
    optimizer=None, gamma=0.999, tau=0.01, unsafe_weight=2.0, pairwise_loss_coeff=0.1
):
    is_train = optimizer is not None
    value_model.train() if is_train else value_model.eval()
    target_value_model.eval()

    total_loss = total_pairwise_loss = total_final_loss = 0.0
    counted_batches = 0

    for hidden_states, labels, mask, scores in tqdm(dataloader, desc="Train" if is_train else "Test"):
        hidden_states = hidden_states.to(device)          # (B, T, D)
        mask = mask.to(device)                            # (B, T) 0/1
        scores = scores.to(device)                        # (B, T) assumed in [-0.5, 0.5]
        labels_tensor = labels.float().to(device)         # (B,) in [-0.5, +0.5]

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        B, T, D = hidden_states.shape

        # Forward (mixed precision OK)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            preds = value_model(hidden_states.view(-1, D))        # (B*T, 1)
            preds = preds.view(B, T).contiguous()                 # (B, T)

        # Target network forward (no grad)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            target_preds = target_value_model(hidden_states.view(-1, D)).view(B, T).contiguous()  # (B, T)

        # Valid transitions t -> t+1
        if T < 2:
            continue
        valid_mask = (mask[:, :-1] * mask[:, 1:]).bool()          # (B, T-1)

        # Soft-Bellman target with min BRT approach
        # Use scores[:, :-1] as immediate "reward" proxy; clamp for safety
        r_t = scores[:, :-1]
        v_next = target_preds[:, 1:]             # keep sane range if desired
        min_target = torch.minimum(v_next, scores[:, 1:])  # or v_next alone if you prefer pure bootstrapping

        with torch.no_grad():
            bootstrapped = (1.0 - gamma) * r_t + gamma * min_target   # (B, T-1)

        # Select valid transitions
        pred_t = preds[:, :-1][valid_mask]                        # (N_valid,)
        target_t = bootstrapped[valid_mask]                       # (N_valid,)

        # Transition weights: unsafe episodes weighted higher
        episode_weights = torch.where(labels_tensor < 0, unsafe_weight, 1.0)  # (B,)
        trans_w = (valid_mask * episode_weights.unsqueeze(1)).float()[valid_mask]  # (N_valid,)

        # Pairwise (temporal) loss normalized by number of valid transitions
        if pred_t.numel() > 0:
            pairwise_mse = F.mse_loss(pred_t.float(), target_t.float(), reduction="none")
            pairwise_loss = (pairwise_mse * trans_w).sum() / trans_w.sum()
        else:
            pairwise_loss = torch.tensor(0.0, device=device)
        
        pairwise_loss_coeff_epoch = pairwise_loss_coeff
        if epoch < warmup_epochs:
            pairwise_loss_coeff_epoch = pairwise_loss_coeff * epoch / warmup_epochs
        
        # Final-state regression (t = last observed token)
        last_indices = mask.sum(dim=1).long() - 1
        last_indices = torch.clamp(last_indices, min=0)
        batch_idx = torch.arange(B, device=device)
        last_preds = preds[batch_idx, last_indices]                # (B,)

        final_w = torch.where(labels_tensor < 0, unsafe_weight, 1.0)  # (B,)
        final_mse = F.mse_loss(last_preds.float(), labels_tensor, reduction="none")
        final_loss = (final_mse * final_w).sum() / final_w.sum()

        loss = pairwise_loss_coeff_epoch * pairwise_loss + final_loss

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(value_model.parameters(), 1.0)
            optimizer.step()
            
            # Soft update target
            with torch.no_grad():
                for tgt, src in zip(target_value_model.parameters(), value_model.parameters()):
                    tgt.data.lerp_(src.data, tau)

        total_loss += loss.item()
        total_pairwise_loss += pairwise_loss.item()
        total_final_loss += final_loss.item()
        counted_batches += 1

    if counted_batches == 0:
        return 0.0, 0.0, 0.0

    return (total_loss / counted_batches,
            total_pairwise_loss / counted_batches,
            total_final_loss / counted_batches)

def train_epoch(value_model, target_value_model, epoch, warmup_epochs, dataloader, optimizer, device, gamma, tau, unsafe_weight, pairwise_loss_coeff):
    return _run_epoch(value_model, target_value_model, epoch, warmup_epochs, dataloader, device, optimizer, gamma=gamma, tau=tau, unsafe_weight=unsafe_weight, pairwise_loss_coeff=pairwise_loss_coeff)

def test_epoch(value_model, target_value_model, epoch, warmup_epochs, dataloader, device, gamma, tau, unsafe_weight, pairwise_loss_coeff):
    return _run_epoch(value_model, target_value_model, epoch, warmup_epochs, dataloader, device, gamma=gamma, tau=tau, unsafe_weight=unsafe_weight, pairwise_loss_coeff=pairwise_loss_coeff)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='mistralai/Ministral-8B-Instruct-2410')
    parser.add_argument('--dataset_name', type=str, default='beavertails')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--anneal_epochs', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--hidden_dims', type=eval, default=[16384, 64])
    parser.add_argument('--gamma_start', type=float, default=0.999)
    parser.add_argument('--gamma_end', type=float, default=0.999)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--value_model_path', type=str, required=True)
    parser.add_argument('--unsafe_weight', type=float, default=2.0)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--pairwise_loss_coeff', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default=directories.DATA_CACHE_DIR)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    wandb.config.update(args)

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    def load_list_tensors(split):
        if "beavertails" in args.dataset_name:
            split = "330k_" + split
        prefix = f'{args.data_dir}/{args.dataset_name}/{args.model_name.replace("/", "_")}/{split}'
        embeddings = torch.load(f'{prefix}_embeddings.pt')
        masks = torch.load(f'{prefix}_masks.pt')
        labels = torch.load(f'{prefix}_final_scores.pt')
        if args.model_name in {"Qwen/Qwen2-1.5B", "meta-llama/Llama-2-7b-hf"}:
            labels = torch.cat([
                (label if torch.is_tensor(label) else torch.tensor(label)).reshape(-1)
                for label in labels
            ])
        scores = torch.load(f'{prefix}_prefix_scores.pt')
        return embeddings, labels, masks, scores

    train_embeds, train_labels, train_masks, train_scores = load_list_tensors("train")
    test_embeds, test_labels, test_masks, test_scores = load_list_tensors("test")

    train_dataset = VariableLengthDataset(train_embeds, train_labels, train_masks, train_scores)
    test_dataset = VariableLengthDataset(test_embeds, test_labels, test_masks, test_scores)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    input_dim = train_embeds[0].shape[-1]
    
    start_epoch = 0
    value_model = value_function.ValueFunction(input_dim=input_dim, hidden_dims=args.hidden_dims)
    value_model.load_state_dict(torch.load(args.value_model_path))
    value_model = value_model.to(device, dtype=torch.bfloat16)
    
    optimizer = optim.Adam(value_model.parameters(), lr=args.lr, weight_decay=0.00001)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    target_value_model = value_function.ValueFunction(input_dim=input_dim, hidden_dims=args.hidden_dims)
    target_value_model.load_state_dict(torch.load(args.value_model_path))
    target_value_model = target_value_model.to(device, dtype=torch.bfloat16)

    model_dir = os.path.join(directories.TRAINED_VALUE_FUNCTION_MODELS_DIR, 'brt_value_function_l_function_as_reward_fp32_embeddings')
    os.makedirs(model_dir, exist_ok=True)
    hidden_dims_str = "_".join(map(str, args.hidden_dims))
    print(f"Training model with hidden dims {hidden_dims_str}")

    gamma_scheduler = GammaScheduler(gamma_start=args.gamma_start, gamma_end=args.gamma_end, anneal_epochs=args.anneal_epochs)
    for epoch in tqdm(range(start_epoch, args.epochs), desc="Training Loop"):
        gamma = gamma_scheduler.get_gamma(epoch)
        train_loss, train_pairwise_loss, train_final_loss = train_epoch(value_model, target_value_model, epoch, args.warmup_epochs, train_loader, optimizer, device, gamma=gamma, tau=args.tau, unsafe_weight=args.unsafe_weight, pairwise_loss_coeff=args.pairwise_loss_coeff)
        test_loss, test_pairwise_loss, test_final_loss = test_epoch(value_model, target_value_model, epoch, args.warmup_epochs, test_loader, device, gamma=gamma, tau=args.tau, unsafe_weight=args.unsafe_weight, pairwise_loss_coeff=args.pairwise_loss_coeff)

        print(f"Epoch {epoch+1}, Gamma: {gamma:.4f}, Train Loss: {train_loss:.4f}, Train Pairwise Loss: {train_pairwise_loss:.4f}, Train Final Loss: {train_final_loss:.4f}, Test Loss: {test_loss:.4f}, Test Pairwise Loss: {test_pairwise_loss:.4f}, Test Final Loss: {test_final_loss:.4f}")
        wandb.log({
            "Training Loss": train_loss,
            "Train Pairwise Loss": train_pairwise_loss,
            "Train Final Loss": train_final_loss,
            "Test Loss": test_loss,
            "Test Pairwise Loss": test_pairwise_loss,
            "Test Final Loss": test_final_loss,
            "Epoch": epoch+1,
            "LR": scheduler.get_last_lr()[0]
        })

        scheduler.step()

        if (epoch + 1) % 1 == 0:
            epoch_file_path = f'{model_dir}/value_model_{args.model_name.replace("/", "_")}_{args.dataset_name}_{args.lr}_batch{args.batch_size}_hidden{hidden_dims_str}_gamma_start{gamma_scheduler.gamma_start}_gamma_end{gamma_scheduler.gamma_end}_anneal{args.anneal_epochs}_tau{args.tau}_unsafe_weight{args.unsafe_weight}_warmup_epochs{args.warmup_epochs}_pairwise_loss_coeff{args.pairwise_loss_coeff}_seed{args.seed}_epoch{epoch+1}.pth'
            torch.save(value_model.state_dict(), epoch_file_path)
            print(f"Saved model to {epoch_file_path}")
            
            prediction_success, prediction_token_locations, all_preds_list, all_masks_list, all_labels_list, first_token_prediction_success_rate, _ = mlp_evaluate_model.evaluate(value_model, test_loader, device, zero_level_label=True)
            print(f"Test Classification Success Rate: {(prediction_success.correct_safe + prediction_success.correct_unsafe) / (prediction_success.correct_safe + prediction_success.correct_unsafe + prediction_success.incorrect_safe + prediction_success.incorrect_unsafe)}")
            print(f"Average Prediction Token Location: {np.mean(prediction_token_locations)} +/- {np.std(prediction_token_locations) / np.sqrt(len(prediction_token_locations))}")
            print(f"Length of prediction_token_locations: {len(prediction_token_locations)}")
            print(f"prediction_success: {prediction_success}")
            print(f"first_token_prediction_success_rate: {first_token_prediction_success_rate}")
            print(f"First Token Test Classification Success Rate: {(first_token_prediction_success_rate.correct_safe + first_token_prediction_success_rate.correct_unsafe) / (first_token_prediction_success_rate.correct_safe + first_token_prediction_success_rate.correct_unsafe + first_token_prediction_success_rate.incorrect_safe + first_token_prediction_success_rate.incorrect_unsafe)}")            

if __name__ == "__main__":
    main()
