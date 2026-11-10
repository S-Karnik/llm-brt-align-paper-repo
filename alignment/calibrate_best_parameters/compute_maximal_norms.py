import json
import os
import torch

import directories

MODEL_NAMES = [
    "Qwen/Qwen2-1.5B",
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Ministral-8B-Instruct-2410",
    "openai/gpt-oss-20b",
    "tiiuae/falcon-7b-instruct"
]

def main():
    max_norms = {}
    for model_name in MODEL_NAMES:
        embeddings_path = os.path.join(
            directories.DATA_CACHE_DIR,
            "beavertails_llm_generated",
            model_name.replace('/', '_'),
            "330k_train_embeddings.pt",
        )
        embeddings = torch.load(embeddings_path)
        stacked_embeddings = torch.cat(embeddings, dim=0)    
        max_norms[model_name] = torch.norm(stacked_embeddings, dim=1).max().item()
        print(f"Max norm for {model_name}: {max_norms[model_name]}")

    with open("max_norms.json", "w") as f:
        json.dump(max_norms, f)

if __name__ == "__main__":
    main()
