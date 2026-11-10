import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader, Dataset
import wandb
import os
import random
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

import directories
from value_function_lib import value_function
from evaluation import evaluate_mlp_model
from toxicity_value_function import data_loading_helpers

os.environ['WANDB_START_METHOD'] = 'thread'
wandb.init(project="llm-control", name="train_value_bellman")

class VariableLengthDataset(Dataset):
    def __init__(self, embeddings, labels, masks):
        self.embeddings = embeddings
        self.labels = labels
        self.masks = masks

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.masks[idx]

def _run_epoch(value_model, dataloader, device, optimizer=None, unsafe_weight=2.0):
    is_train = optimizer is not None
    value_model.train() if is_train else value_model.eval()
    total_loss, count_batches = 0, 0

    for hidden_states, labels, mask in tqdm(dataloader, desc="Train" if is_train else "Test"):
        hidden_states = hidden_states.to(device)
        labels = labels.to(device, dtype=torch.bfloat16)
        mask = mask.to(device)
        
        if is_train:
            optimizer.zero_grad()

        B, T, D = hidden_states.shape
        
        if T == 0:
            continue
                
        labels_tensor = labels.float().to(device)
        all_preds = value_model(hidden_states.to(torch.bfloat16).view(-1, D)).view(B, T).to(torch.float32)
        targets = torch.repeat_interleave(labels_tensor.to(torch.float32), mask.sum(dim=1))
        weights = torch.where(targets < 0, unsafe_weight, 1.0)
        squared_errors = F.mse_loss(all_preds[mask.bool()].squeeze(-1).float(), targets, reduction='none')
        
        target_loss = (weights * squared_errors).sum()
                        
        if is_train:
            target_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += target_loss.item()
        count_batches += 1

    return total_loss / count_batches

def train_epoch(value_model, dataloader, optimizer, device, unsafe_weight):
    return _run_epoch(value_model, dataloader, device, optimizer=optimizer, unsafe_weight=unsafe_weight)

def test_epoch(value_model, dataloader, device, unsafe_weight):
    return _run_epoch(value_model, dataloader, device, unsafe_weight=unsafe_weight)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='mistralai/Ministral-8B-Instruct-2410', choices=[
        "mistralai/Ministral-8B-Instruct-2410", 
        "Qwen/Qwen2-1.5B", 
        "meta-llama/Llama-2-7b-hf", 
        "openai/gpt-oss-20b", 
        "tiiuae/falcon-7b-instruct"
    ])
    parser.add_argument('--dataset_name', type=str, default='beavertails')
    parser.add_argument('--data_dir', type=str, default=directories.DATA_CACHE_DIR)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--hidden_dims', type=eval, default=[16384, 64])
    parser.add_argument('--value_model_path', type=str, default=None)
    parser.add_argument('--unsafe_weight', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_stochastic_embeddings', action='store_true')
    
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
        data_dir = args.data_dir
        if args.use_stochastic_embeddings:
            data_dir = data_dir + "/stochastic"
        prefix = f'{data_dir}/{args.dataset_name}/{args.model_name.replace("/", "_")}/{split}'        
        embeddings = torch.load(f'{prefix}_embeddings.pt')
        scores = [torch.tensor(score) for score in torch.load(f'{prefix}_scores.pt')]
        masks = torch.load(f'{prefix}_masks.pt')
        is_safes = torch.load(f'{prefix}_is_safes.pt')
        return embeddings, scores, masks

    train_embeds, train_labels, train_masks = load_list_tensors("train")
    test_embeds, test_labels, test_masks = load_list_tensors("test")

    train_dataset = VariableLengthDataset(train_embeds, train_labels, train_masks)
    test_dataset = VariableLengthDataset(test_embeds, test_labels, test_masks)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_loading_helpers.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_loading_helpers.collate_fn)

    input_dim = train_embeds[0].shape[-1]
    value_model = value_function.ValueFunction(input_dim=input_dim, hidden_dims=args.hidden_dims)
    if args.value_model_path is not None:
        value_model.load_state_dict(torch.load(args.value_model_path))
    optimizer = optim.Adam(value_model.parameters(), lr=args.lr, weight_decay=0.00001)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    value_model = value_model.to(dtype=torch.bfloat16).to(device)

    parent_dir_name = 'sample_based_brt_value_function_fp32_embeddings'
    if args.use_stochastic_embeddings:
        parent_dir_name = "stochastic_" + parent_dir_name

    model_dir = os.path.join(directories.TRAINED_VALUE_FUNCTION_MODELS_DIR, parent_dir_name)
    os.makedirs(model_dir, exist_ok=True)
    
    hidden_dims_str = "_".join(map(str, args.hidden_dims))
    print(f"Training model with hidden dims {hidden_dims_str}")

    for epoch in tqdm(range(args.epochs), desc="Training Loop"):
        train_loss = train_epoch(value_model, train_loader, optimizer, device, unsafe_weight=args.unsafe_weight)
        test_loss = test_epoch(value_model, test_loader, device, unsafe_weight=args.unsafe_weight)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        wandb.log({
            "Training Loss": train_loss,
            "Test Loss": test_loss,
            "Epoch": epoch+1,
            "LR": scheduler.get_last_lr()[0]
        })

        scheduler.step()

        if (epoch + 1) % 1 == 0:
            epoch_file_path = f'{model_dir}/value_model_{args.model_name.replace("/", "_")}_{args.dataset_name}_{args.lr}_batch{args.batch_size}_hidden{hidden_dims_str}_unsafe_weight{args.unsafe_weight}_seed{args.seed}_epoch{epoch+1}.pth'
            torch.save(value_model.state_dict(), epoch_file_path)
            print(f"Saved model to {epoch_file_path}")
            
            prediction_success, prediction_token_locations, all_preds_list, all_masks_list, all_labels_list, first_token_prediction_success_rate, _ = evaluate_mlp_model.evaluate(value_model, test_loader, device, zero_level_label=True)
            print(f"Test Classification Success Rate: {(prediction_success.correct_safe + prediction_success.correct_unsafe) / (prediction_success.correct_safe + prediction_success.correct_unsafe + prediction_success.incorrect_safe + prediction_success.incorrect_unsafe)}")
            print(f"Average Prediction Token Location: {np.mean(prediction_token_locations)} +/- {np.std(prediction_token_locations) / np.sqrt(len(prediction_token_locations))}")
            print(f"Length of prediction_token_locations: {len(prediction_token_locations)}")
            print(f"prediction_success: {prediction_success}")
            print(f"first_token_prediction_success_rate: {first_token_prediction_success_rate}")
            print(f"First Token Test Classification Success Rate: {(first_token_prediction_success_rate.correct_safe + first_token_prediction_success_rate.correct_unsafe) / (first_token_prediction_success_rate.correct_safe + first_token_prediction_success_rate.correct_unsafe + first_token_prediction_success_rate.incorrect_safe + first_token_prediction_success_rate.incorrect_unsafe)}")            

if __name__ == "__main__":
    main()
