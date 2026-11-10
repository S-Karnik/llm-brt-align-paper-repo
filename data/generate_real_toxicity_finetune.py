import os
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification
import directories

INPUT_DIM_MAP = {
    "mistralai/Ministral-8B-Instruct-2410": 4096,
    "meta-llama/Llama-2-7b-hf": 4096,
    "Qwen/Qwen2-1.5B": 1536,
    "EleutherAI/pythia-6.9b": 4096,   
    "tiiuae/falcon-7b-instruct": 4096,
}

NO_CHAT_TEMPLATE_MODELS = {"Qwen/Qwen2-1.5B"}

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--layer_idx", type=int, default=20)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=12)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.shard_id % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    save_dir = f"{directories.DATA_CACHE_DIR}/real_toxicity_llm_generated_layer_{args.layer_idx}/{args.model_name.replace('/', '_')}"
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    generator = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32 if args.model_name not in ["openai/gpt-oss-20b", "tiiuae/falcon-7b-instruct"] else torch.bfloat16,
        trust_remote_code=True
    ).eval().to(device)

    # Load RoBERTa classifier for offensive detection
    clf_model_name = "cardiffnlp/twitter-roberta-base-offensive"
    classifier = AutoModelForSequenceClassification.from_pretrained(clf_model_name).eval().to(device)
    clf_tokenizer = AutoTokenizer.from_pretrained(clf_model_name)

    raw_dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
    if (
        args.model_name not in NO_CHAT_TEMPLATE_MODELS
        and getattr(tokenizer, "chat_template", None) is not None
    ):
        prompts = [tokenizer.apply_chat_template([{"role": "user", "content": ex["prompt"]["text"]}], tokenize=False, add_generation_prompt=True) for ex in raw_dataset if ex["prompt"]["toxicity"] is not None]
    else:
        prompts = [ex["prompt"]["text"] for ex in raw_dataset if ex["prompt"]["toxicity"] is not None]
    total_len = len(prompts)
    per_shard = (total_len + args.num_shards - 1) // args.num_shards
    start_idx = args.shard_id * per_shard
    end_idx = min(start_idx + per_shard, total_len)
    subset = prompts[start_idx:end_idx]

    batch_size = args.batch_size
    max_new_tokens = 128
    max_length = 512

    embeddings, masks, labels, all_responses, all_prompts, all_scores = [], [], [], [], [], []

    for batch_start in tqdm(range(0, len(subset), batch_size), desc=f"Shard {args.shard_id}"):
        batch_prompts = subset[batch_start:batch_start + batch_size]

        # ----- Generation (no gradients) -----
        input_enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        with torch.no_grad():
            outputs = generator.generate(
                input_ids=input_enc.input_ids,
                attention_mask=input_enc.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
                use_cache=False if args.model_name in ["tiiuae/falcon-7b-instruct"] else True
            )

        sequences = outputs.sequences
        gen_sequences = sequences[:, input_enc.input_ids.shape[1]:]
        responses = [r.strip() for r in tokenizer.batch_decode(gen_sequences, skip_special_tokens=True)]

        full_texts = [p + "\n" + r for p, r in zip(batch_prompts, responses)]

        # free gen intermediates before the heavy featurizer pass
        del outputs, sequences, gen_sequences
        torch.cuda.empty_cache()

        # ----- Featurizer pass (same model), no KV cache, shortest padding -----
        enc = tokenizer(
            full_texts,
            return_tensors="pt",
            padding="longest",          # <---- key change (avoid padding to 512)
            truncation=True,
            max_length=max_length
        ).to(device)

        with torch.no_grad():
            # temporarily disable KV cache for this forward to cut memory
            orig_use_cache = generator.config.use_cache
            generator.config.use_cache = False
            feats = generator(                     # same model
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                output_hidden_states=True,
            )
            generator.config.use_cache = orig_use_cache

        h = feats.hidden_states[args.layer_idx]  # (B, T, D)

        for i in range(enc.input_ids.size(0)):
            input_id = enc.input_ids[i]
            h_i = h[i]
            valid_len = enc.attention_mask[i].sum().item()  # robust valid length
            h_i = h_i[enc.attention_mask[i].bool()].to(torch.float32).cpu()

            prompt_len = tokenizer(batch_prompts[i], return_tensors="pt", truncation=True).input_ids.shape[1]
            response_mask = torch.zeros(valid_len, dtype=torch.long)
            if prompt_len < valid_len:
                response_mask[prompt_len:] = 1

            if response_mask.sum() == 0:
                continue

            # classifier stays as-is
            with torch.no_grad():
                clf_input = clf_tokenizer(
                    responses[i],
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                ).to(device)
                logits = classifier(**clf_input).logits
                score = torch.softmax(logits, dim=-1)[0][1].item()
            label = int(score < 0.5)

            embeddings.append(h_i[response_mask.bool()])
            masks.append(response_mask)
            labels.append(label)
            all_responses.append(responses[i])
            all_prompts.append(batch_prompts[i])
            all_scores.append(score)

            del h_i

        del enc, feats, h, input_enc
        torch.cuda.empty_cache()

    torch.save(all_responses, os.path.join(save_dir, f"train_responses_shard{args.shard_id}.pt"))
    torch.save(embeddings, os.path.join(save_dir, f"train_embeddings_shard{args.shard_id}.pt"))
    torch.save(masks, os.path.join(save_dir, f"train_masks_shard{args.shard_id}.pt"))
    torch.save(labels, os.path.join(save_dir, f"train_labels_shard{args.shard_id}.pt"))
    torch.save(all_prompts, os.path.join(save_dir, f"train_prompts_shard{args.shard_id}.pt"))
    torch.save(all_scores, os.path.join(save_dir, f"train_scores_shard{args.shard_id}.pt"))

    print(f"[Shard {args.shard_id}] Saved outputs to {save_dir}")

if __name__ == "__main__":
    main()
