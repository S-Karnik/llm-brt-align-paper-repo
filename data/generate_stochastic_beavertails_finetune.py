import os
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch.nn.functional as F
import directories

INPUT_DIM_MAP = {
    "mistralai/Ministral-8B-Instruct-2410": 4096,
    "meta-llama/Llama-2-7b-hf": 4096,
    "Qwen/Qwen2-1.5B": 1536,
    "EleutherAI/pythia-6.9b": 4096,
    "tiiuae/falcon-7b-instruct": 4096,
}

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--layer_idx", type=int, default=20)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=12)

    # stochastic sampling controls
    parser.add_argument("--num_samples_per_prompt", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)

    # seed for reproducibility
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.shard_id % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    # base seed depends on global seed and shard so that each shard is different
    base_seed = args.seed
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed_all(base_seed)

    save_dir = f"{directories.DATA_CACHE_DIR}/beavertails_llm_generated/{args.model_name.replace('/', '_')}"
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    generator = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32 if args.model_name not in ["openai/gpt-oss-20b", "tiiuae/falcon-7b-instruct"] else torch.bfloat16,
        trust_remote_code=True
    ).eval().to(device)
    
    # Load final token classifier: cardiffnlp/twitter-roberta-base-offensive
    clf_model_name = "cardiffnlp/twitter-roberta-base-offensive"
    classifier = AutoModelForSequenceClassification.from_pretrained(clf_model_name).eval().to(device)

    dataset = load_dataset("PKU-Alignment/BeaverTails", split=args.split)
    prompts = [ex["prompt"] for ex in dataset]
    total_len = len(prompts)
    per_shard = (total_len + args.num_shards - 1) // args.num_shards
    start_idx = args.shard_id * per_shard
    end_idx = min(start_idx + per_shard, total_len)
    prompt_subset = prompts[start_idx:end_idx]

    batch_size = args.batch_size
    max_new_tokens = 128
    max_length = 512

    embeddings, masks, labels, all_prompts, all_responses, all_scores = [], [], [], [], [], []
    
    clf_tokenizer = AutoTokenizer.from_pretrained(clf_model_name)

    for batch_start in tqdm(range(0, len(prompt_subset), batch_size), desc=f"Shard {args.shard_id}"):
        batch_prompts = prompt_subset[batch_start:batch_start + batch_size]

        # encode prompts once per batch
        input_enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        # draw multiple stochastic samples per prompt
        for sample_idx in range(args.num_samples_per_prompt):
            # per-sample deterministic seed on this shard
            sample_seed = base_seed + batch_start * args.num_samples_per_prompt + sample_idx
            torch.manual_seed(sample_seed)
            torch.cuda.manual_seed_all(sample_seed)

            # Generation (stochastic, no gradients)
            with torch.no_grad():
                outputs = generator.generate(
                    input_ids=input_enc.input_ids,
                    attention_mask=input_enc.attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=False,
                    use_cache=False if args.model_name in ["tiiuae/falcon-7b-instruct"] else True,
                )

            sequences = outputs.sequences
            decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)
            responses = [dec[len(prompt):].strip() for dec, prompt in zip(decoded, batch_prompts)]
            full_texts = [p + "\n" + r for p, r in zip(batch_prompts, responses)]

            # free gen intermediates before the heavy featurizer pass
            del outputs, sequences, decoded
            torch.cuda.empty_cache()

            # Featurizer pass (same model), no KV cache, shortest padding
            enc = tokenizer(
                full_texts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_length
            ).to(device)

            with torch.no_grad():
                orig_use_cache = generator.config.use_cache
                generator.config.use_cache = False
                feats = generator(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    output_hidden_states=True,
                )
                generator.config.use_cache = orig_use_cache

            h = feats.hidden_states[args.layer_idx]  # (B, T, D)

            for i in range(enc.input_ids.size(0)):
                h_i = h[i]
                valid_len = enc.attention_mask[i].sum().item()
                h_i = h_i[enc.attention_mask[i].bool()].to(torch.float32).cpu()
                
                prompt_len = tokenizer(batch_prompts[i], return_tensors="pt", truncation=True).input_ids.shape[1]
                response_mask = torch.zeros(valid_len, dtype=torch.long)
                if prompt_len < valid_len:
                    response_mask[prompt_len:] = 1

                if response_mask.sum() == 0:
                    continue

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

            del enc, feats, h
            torch.cuda.empty_cache()

        del input_enc
        torch.cuda.empty_cache()

    torch.save(all_prompts, os.path.join(save_dir, f"{args.split}_prompts_shard{args.shard_id}.pt"))
    torch.save(all_responses, os.path.join(save_dir, f"{args.split}_responses_shard{args.shard_id}.pt"))
    torch.save(embeddings, os.path.join(save_dir, f"{args.split}_embeddings_shard{args.shard_id}.pt"))
    torch.save(masks, os.path.join(save_dir, f"{args.split}_masks_shard{args.shard_id}.pt"))
    torch.save(labels, os.path.join(save_dir, f"{args.split}_labels_shard{args.shard_id}.pt"))
    torch.save(all_scores, os.path.join(save_dir, f"{args.split}_logits_shard{args.shard_id}.pt"))

    print(f"[Shard {args.shard_id}] Saved outputs to {save_dir}")

if __name__ == "__main__":
    main()
