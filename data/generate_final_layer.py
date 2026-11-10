#!/usr/bin/env python3
import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

BFLOAT16_MODELS = {
    "openai/gpt-oss-20b",
    "tiiuae/falcon-7b-instruct",
}

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--prompts_path", type=str, required=True, help=".pt list[str]")
    p.add_argument("--responses_path", type=str, required=True, help=".pt list[str]")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument(
        "--boundary_mode",
        type=str,
        choices=["legacy", "robust"],
        default="legacy",
        help=(
            "legacy = start response at tokenized(prompt) length (matches your original generators; "
            "newline is included in response). robust = start after tokenized(prompt+'\\n') "
            "and account for special tokens."
        ),
    )
    args = p.parse_args()

    # ---- Device ----
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # ---- Load saved texts ----
    prompts = torch.load(args.prompts_path)
    responses = torch.load(args.responses_path)
    assert len(prompts) == len(responses), "prompts/responses length mismatch"

    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, f"{args.split}_embeddings_lastlayer.pt")

    # ---- Tokenizer & LM ----
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dtype = torch.bfloat16 if args.model_name in BFLOAT16_MODELS else torch.float32
    lm = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, trust_remote_code=True
    ).eval().to(device)

    embeddings = []
    B = args.batch_size

    for i in tqdm(range(0, len(prompts), B), desc="Re-encode last layer"):
        ps = prompts[i:i + B]
        rs = responses[i:i + B]
        full_texts = [p + "\n" + r for p, r in zip(ps, rs)]

        enc = tok(
            full_texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=args.max_length,
        ).to(device)

        # Forward with last-layer hidden states (pre LM head)
        orig_use_cache = lm.config.use_cache
        lm.config.use_cache = False
        out = lm(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            output_hidden_states=True,
        )
        lm.config.use_cache = orig_use_cache
        h_last = out.hidden_states[-1]  # (B, T, D)

        # Per-example: remove padding via boolean mask, then slice response span
        for j in range(h_last.size(0)):
            mask = enc.attention_mask[j].bool()
            h_j = h_last[j][mask].to(torch.float32).cpu()  # (valid_len, D)
            valid_len = h_j.size(0)

            if valid_len == 0:
                d = h_last.size(-1)
                embeddings.append(torch.empty(0, d, dtype=torch.float32))
                continue

            if args.boundary_mode == "legacy":
                # Match your original generators: start at tokenized(prompt),
                # i.e., newline belongs to the "response" segment.
                prompt_len = tok(
                    ps[j], return_tensors="pt", truncation=True, max_length=args.max_length
                ).input_ids.shape[1]
                resp_start = min(max(0, prompt_len), valid_len)
            else:
                # Robust: account for specials and explicit "\n"
                prompt_ids_no_spec = tok(
                    ps[j] + "\n", add_special_tokens=False, return_tensors="pt"
                ).input_ids[0]
                combo_no_spec = tok(
                    full_texts[j], add_special_tokens=False, return_tensors="pt"
                ).input_ids[0]
                specials = valid_len - combo_no_spec.size(0)
                resp_start = max(0, min(specials + prompt_ids_no_spec.size(0), valid_len))

            # Keep only response-token embeddings
            embeddings.append(h_j[resp_start:])  # (n_resp_tokens, D)

        # Free per-batch memory
        del enc, out, h_last
        if device.type == "cuda":
            torch.cuda.empty_cache()

    torch.save(embeddings, save_path)
    print(f"Saved embeddings to: {save_path}  (num_examples={len(embeddings)})")

if __name__ == "__main__":
    main()
