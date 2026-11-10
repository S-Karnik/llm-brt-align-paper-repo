from torch.nn.utils.rnn import pad_sequence
import torch

def pad_and_shift_masks(masks, batch_first=True, padding_value=0):
    if not masks:
        return torch.empty(0)

    padded = pad_sequence(masks, batch_first=True, padding_value=padding_value)
    # Find first nonzero per row (safe even if values aren’t strictly 0/1)
    first_one = (padded != 0).to(torch.int64).argmax(dim=1)  # 0 if all zeros
    B, L = padded.shape

    # Build per-row indices: [s, s+1, ..., s+L-1], then clip and mask
    base = torch.arange(L, device=padded.device).unsqueeze(0).expand(B, L)
    idx = base + first_one.unsqueeze(1)             # shape (B, L)
    valid = idx < L                                  # which positions stay in-bounds
    idx = idx.clamp_max(L - 1)                       # safe gather

    out = torch.gather(padded, dim=1, index=idx)     # rotated-left view via gather
    out = out * valid.to(out.dtype)                  # zero out wrapped positions

    return out if batch_first else out.transpose(0, 1)


def collate_fn(batch):
    embeddings, labels, masks = zip(*batch)
    padded_embeddings = pad_sequence(embeddings, batch_first=True)
    lengths = torch.tensor([emb.shape[0] for emb in embeddings], dtype=torch.long)
    mask_counts = torch.tensor([mask.sum().item() for mask in masks], dtype=torch.long)
    if not torch.equal(mask_counts, lengths):
        raise ValueError("Mask token counts do not match response embedding lengths.")
    padded_masks = torch.arange(padded_embeddings.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    labels = torch.tensor(labels, dtype=torch.float32) - 0.5
    return padded_embeddings, labels, padded_masks
