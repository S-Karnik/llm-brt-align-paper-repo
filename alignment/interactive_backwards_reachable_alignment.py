# backwards_reachable_alignment.py
import argparse
import csv
import json
import hydra
import os
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    set_seed,
)

from datasets import load_dataset
from omegaconf import DictConfig

import os
import sys

from tqdm import tqdm

# Add the project root to the path to ensure consistent imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import directories
from alignment.aligners.base_aligner import (
    AlignConfig,
    LeastRestrictiveReachableAlignerSpec,
    NoiseConfig,
    ReControlAlignerSpec,
    SmoothBlendReachableAlignerSpec,
    GradientAscentReachableAlignerSpec,
    SAPAlignerSpec
)
from alignment.aligners import backwards_reachable_aligner, recontrol_aligner, sap_aligner


NO_CHAT_TEMPLATE_MODELS = {"Qwen/Qwen2-1.5B"}


def format_chat_prompt(tokenizer, model_name: str, prompt: str) -> str:
    """Match the prompt formatting used by generate_beavertails_finetune.py."""
    if (
        model_name not in NO_CHAT_TEMPLATE_MODELS
        and getattr(tokenizer, "chat_template", None) is not None
    ):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


# ----------------------------
# PAIR dataset helper
# ----------------------------
def load_pair_dataset(dataset_name: str):
    """
    Load PAIR jailbreak prompts from JailbreakBench for any model.

    Expects dataset_name of the form:
      'pair_gpt-3.5-turbo-1106'
      'pair_gpt-4-0125-preview'
      'pair_llama-2-7b-chat-hf'
      'pair_vicuna-13b-v1.5'
    """
    assert dataset_name.startswith("pair_"), f"Invalid PAIR dataset_name: {dataset_name}"

    json_path = os.path.join(os.environ.get("PAIR_PROMPTS_DIR", "pair_prompts"), dataset_name + "_prompts.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # list of dicts

    return data

# ----------------------------
# Safety classifier helper
# ----------------------------
def classify_texts(texts: List[str], device: torch.device, clf_tok: AutoTokenizer, clf: nn.Module) -> Tuple[List[int], List[List[float]]]:
    # Cap at 512 to satisfy roberta-base (514 incl. specials)
    cap = 512
    enc = clf_tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cap
    ).to(device)
    with torch.no_grad():
        logits = clf(**enc).logits
        probs = F.softmax(logits, dim=-1)          # [B, 2]  ->  index 1 = offensive
        preds = probs.argmax(dim=-1)               # 0 = safe, 1 = unsafe/offensive
    return preds.cpu().tolist(), probs.cpu().tolist()


# ----------------------------
# CLI / runner
# ----------------------------
def load_hydra_config():
    """Load Hydra config without interfering with command line arguments."""
    # Save and restore sys.argv to prevent Hydra from processing our command line args
    import sys
    saved_argv = sys.argv.copy()
    try:
        # Temporarily replace sys.argv with just the script name
        sys.argv = [saved_argv[0]]
        sap_config_dir = os.environ.get("SAP_CONFIG_DIR")
        if not sap_config_dir:
            raise FileNotFoundError(
                "SaP requires an external Safety Polytope config directory. "
                "Set SAP_CONFIG_DIR to the directory containing eval_config.yaml."
            )
        
        # Initialize Hydra
        if os.path.isabs(sap_config_dir):
            hydra.initialize_config_dir(version_base="1.1", config_dir=sap_config_dir)
        else:
            hydra.initialize(version_base="1.1", config_path=sap_config_dir)
        
        # Load the config
        cfg = hydra.compose(config_name="eval_config")
        return cfg
    finally:
        # Restore original command line arguments
        sys.argv = saved_argv

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2-1.5B")
    p.add_argument("--layer_idx", type=int, default=20)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--epsilon_search", type=float, default=0.0)
    p.add_argument("--epsilon_cap", type=float, default=0.0)
    p.add_argument("--num_steer_samples", type=int, default=32)
    p.add_argument("--value_binary_search_max_iter", type=int, default=2)
    p.add_argument("--norm", type=str, choices=["l2", "linf"], default="l2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--value_hidden_dims", type=eval, default=[16384, 64])
    p.add_argument("--model_ckpt", type=str, default=None)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--aligner_type", type=str, choices=["brt_least_restrictive", "brt_blend", "brt_gradient_ascent", "sap", "recontrol"], required=True)
    p.add_argument("--split_name", type=str, default="train")
    p.add_argument("--value_threshold", type=float, default=0.2)
    p.add_argument("--value_coeff_gamma", type=float, default=0.1)
    p.add_argument("--calibration", action="store_true")
    
    # Dataset-specific args
    p.add_argument("--dataset_name", type=str, default="real_toxicity")
    p.add_argument("--sample_dataset", type=int, default=None)
    p.add_argument("--only_unsafe", action="store_true")
    
    # dataset / IO
    p.add_argument("--prompt", type=str, default="")  # used if not dataset mode
    p.add_argument(
        "--no_save",
        action="store_true",
        help="Print generated responses without writing result files.",
    )
    p.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply the target model's chat template to a single --prompt.",
    )
    p.add_argument(
        "--xstest_prompts_path",
        type=str,
        default=os.environ.get(
            "XSTEST_PROMPTS_PATH",
            os.environ.get("XSTEST_PROMPTS_CSV", "xstest/xstest_prompts.csv"),
        ),
        help="Local path to the official XSTest prompt CSV.",
    )

    # Recontrol-specific args
    p.add_argument("--num_recontrol_updates", type=int, default=10)
    p.add_argument("--recontrol_step_size", type=float, default=0.01)

    # SafetyPolytope-specific args
    p.add_argument("--sap_lambda_weight", type=float, default=1.0)
    p.add_argument("--sap_safe_violation_weight", type=float, default=0.0001)
    p.add_argument("--sap_steer_first_n_tokens", type=int, default=20)

    p.add_argument("--brt_type", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    
    args = p.parse_args()
        
    # Only load hydra config if using SAP aligner
    sap_cfg = None
    if args.aligner_type == "sap":
        sap_cfg = load_hydra_config()

    aligner_spec = None
    
    save_descriptor = ""

    if args.aligner_type == "brt_least_restrictive":
        aligner_spec = LeastRestrictiveReachableAlignerSpec(
            noise_cfg=NoiseConfig(
                radius_mode="uniform_volume",
                epsilon_search=args.epsilon_search,
                epsilon_cap=args.epsilon_cap,
                K=args.num_steer_samples,
                norm=args.norm,
                include_base=False,
            ),
            value_model_ckpt=args.model_ckpt,
            value_hidden_dims=args.value_hidden_dims,
            value_threshold=args.value_threshold,
        )
        save_descriptor = f"brt_least_restrictive_value_threshold_{args.value_threshold}_epsilon_search_{args.epsilon_search}_epsilon_cap_{args.epsilon_cap}"

        if args.epsilon_search == 0:
            save_descriptor = "base_llm"
    elif args.aligner_type == "brt_blend":
        aligner_spec = SmoothBlendReachableAlignerSpec(
            noise_cfg=NoiseConfig(
                radius_mode="uniform_volume",
                epsilon_search=args.epsilon_search,
                epsilon_cap=args.epsilon_cap,
                K=args.num_steer_samples,
                norm=args.norm,
                include_base=True,
            ),
            value_model_ckpt=args.model_ckpt,
            value_hidden_dims=args.value_hidden_dims,
            value_coeff_gamma=args.value_coeff_gamma,
            value_binary_search_max_iter=args.value_binary_search_max_iter,
        )
        save_descriptor = f"brt_blend_value_coeff_gamma_{args.value_coeff_gamma}_epsilon_search_{args.epsilon_search}_epsilon_cap_{args.epsilon_cap}_value_binary_search_max_iter_{args.value_binary_search_max_iter}"
    elif args.aligner_type == "brt_gradient_ascent":
        aligner_spec = GradientAscentReachableAlignerSpec(
            num_updates=args.num_recontrol_updates,
            step_size=args.recontrol_step_size,
            value_model_ckpt=args.model_ckpt,
            value_hidden_dims=args.value_hidden_dims,
            value_threshold=args.value_threshold,
        )
        save_descriptor = f"brt_gradient_ascent_num_updates_{args.num_recontrol_updates}_step_size_{args.recontrol_step_size}"
    elif args.aligner_type == "sap":
        # Update the config with command line arguments
        if hasattr(sap_cfg, 'model_path') or 'model_path' in sap_cfg:
            sap_cfg.model_path = args.model_name
        if hasattr(sap_cfg, 'polytope_weight_path') or 'polytope_weight_path' in sap_cfg:
            sap_cfg.polytope_weight_path = args.model_ckpt
        if hasattr(sap_cfg, 'steer_layer') or 'steer_layer' in sap_cfg:
            sap_cfg.steer_layer = args.layer_idx
        if hasattr(sap_cfg, 'lambda_weight') or 'lambda_weight' in sap_cfg:
            sap_cfg.lambda_weight = args.sap_lambda_weight
        if hasattr(sap_cfg, 'safe_violation_weight') or 'safe_violation_weight' in sap_cfg:
            sap_cfg.safe_violation_weight = args.sap_safe_violation_weight
        if hasattr(sap_cfg, 'steer_first_n_tokens') or 'steer_first_n_tokens' in sap_cfg:
            sap_cfg.steer_first_n_tokens = args.sap_steer_first_n_tokens
        # Important: let's set the use_safe_rep_model to True for SAP aligner
        sap_cfg.use_safe_rep_model = True
        aligner_spec = SAPAlignerSpec(cfg=sap_cfg)
        if (
            args.calibration
            or args.sap_lambda_weight != 1.0
            or args.sap_safe_violation_weight != 0.0001
            or args.sap_steer_first_n_tokens != 20
        ):
            save_descriptor = (
                f"sap_lambda_weight_{args.sap_lambda_weight}_"
                f"safe_violation_weight_{args.sap_safe_violation_weight}_"
                f"steer_first_n_tokens_{args.sap_steer_first_n_tokens}"
            )
    elif args.aligner_type == "recontrol":
        aligner_spec = ReControlAlignerSpec(
            num_updates=args.num_recontrol_updates,
            step_size=args.recontrol_step_size,
            value_model_ckpt=args.model_ckpt,
            value_hidden_dims=args.value_hidden_dims,
        )
        save_descriptor = f"recontrol_num_updates_{args.num_recontrol_updates}_step_size_{args.recontrol_step_size}"
    else:
        raise ValueError(f"Unknown aligner type: {args.aligner_type}")
    
    if "brt" in args.aligner_type:
        assert args.brt_type is not None

    cfg = AlignConfig(
        dataset_name=args.dataset_name,
        split_name=args.split_name,
        model_name=args.model_name,
        layer_idx=args.layer_idx,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        device=args.device,
        bf16=False,
        num_shards=args.num_shards,
        shard_idx=args.shard_idx,
        aligner_spec=aligner_spec,
        save_descriptor=save_descriptor,
        sample_dataset=args.sample_dataset,
        only_unsafe=args.only_unsafe,
        calibration=args.calibration,
        brt_type=args.brt_type,
    )
    return cfg, args

def main():
    cfg, args = parse_args()
    set_seed(cfg.seed)

    # ----------------------------
    # Aligner setup
    # ----------------------------
    if isinstance(cfg.aligner_spec, LeastRestrictiveReachableAlignerSpec):
        aligner = backwards_reachable_aligner.LeastRestrictiveReachableAligner(cfg)
    elif isinstance(cfg.aligner_spec, SmoothBlendReachableAlignerSpec):
        aligner = backwards_reachable_aligner.SmoothBlendReachableAligner(cfg)
    elif isinstance(cfg.aligner_spec, SAPAlignerSpec):
        aligner = sap_aligner.SAPAligner(cfg)
    elif isinstance(cfg.aligner_spec, ReControlAlignerSpec):
        aligner = recontrol_aligner.ReControlAligner(cfg)
    elif isinstance(cfg.aligner_spec, GradientAscentReachableAlignerSpec):
        aligner = backwards_reachable_aligner.GradientAscentReachableAligner(cfg)
    else:
        raise ValueError(f"Unknown aligner spec type: {type(cfg.aligner_spec)}")

    processed_model_name = cfg.model_name.replace('/', '_')

    # ----------------------------
    # Classifier
    # ----------------------------
    clf_model_name = "cardiffnlp/twitter-roberta-base-offensive"
    classifier_tokenizer = AutoTokenizer.from_pretrained(clf_model_name)
    classifier = AutoModelForSequenceClassification.from_pretrained(clf_model_name).eval().to(cfg.device)

    # ----------------------------
    # Save dir
    # ----------------------------
    sample_suffix = f"_sampled_{cfg.sample_dataset}" if cfg.sample_dataset is not None else ""
    calibration_str = "" if not cfg.calibration else "/calibration"
    method_str = args.aligner_type 
    if cfg.brt_type is not None:
        method_str = cfg.brt_type + "/" + method_str
    unsafe_only = "" if not args.only_unsafe else "_only_unsafe"
    save_dir = os.path.join(
        directories.DATA_CACHE_DIR,
        "alignment_results",
        f"seed_{args.seed}",
        processed_model_name,
        cfg.dataset_name,
        f"{cfg.split_name}{unsafe_only}{sample_suffix}{calibration_str}",
        method_str,
        cfg.save_descriptor,
    )
    if not args.no_save:
        print(f"Saving to {save_dir}", flush=True)
        os.makedirs(save_dir, exist_ok=True)

    prompts = []
    saved_prompts = None
    categories = []
    if args.prompt != "":
        # Single prompt mode
        print(f"Using single prompt: {args.prompt}")
        saved_prompts = [args.prompt]
        prompts = [
            format_chat_prompt(aligner.tokenizer, cfg.model_name, args.prompt)
            if args.apply_chat_template
            else args.prompt
        ]
        labels = [None]
    else:
        dataset_name = args.dataset_name
        if dataset_name == "XSTest":
            if not os.path.exists(args.xstest_prompts_path):
                raise FileNotFoundError(
                    f"Official XSTest prompts not found: {args.xstest_prompts_path}"
                )
            with open(args.xstest_prompts_path, newline="", encoding="utf-8") as f:
                xstest_rows = list(csv.DictReader(f))
            raw_prompts = [row["prompt"] for row in xstest_rows]
            saved_prompts = ["Question: " + prompt + "\nAnswer:" for prompt in raw_prompts]
            prompts = [
                format_chat_prompt(aligner.tokenizer, cfg.model_name, prompt)
                for prompt in raw_prompts
            ]
            labels = [row["type"] for row in xstest_rows]
        elif dataset_name.startswith("pair_"):
            jailbreak_data = load_pair_dataset(dataset_name)
            prompts = []
            labels = []
            categories = []
            for ex in jailbreak_data:
                if ex['prompt'] is not None:
                    prompts.append(ex['prompt'])
                else:
                    prompts.append(ex['goal'])
                labels.append(ex['jailbroken'])
                categories.append(ex.get('category', 'unknown'))
        elif dataset_name == "safe_decoder_attackers":
            assert cfg.model_name == "meta-llama/Llama-2-7b-hf", "SafeDecoding-Attackers dataset only supported for Llama-2-7b-hf"
            full_safe_decoder_attackers_dataset = load_dataset("UWNSL/SafeDecoding-Attackers")
            prompts = []
            labels = []
            categories = []
            for row in full_safe_decoder_attackers_dataset['train']:
                if row['target-model'] == "llama2":
                    from fastchat.conversation import get_conv_template
                    
                    # conv = get_conv_template("llama-2")
                    # conv.append_message(conv.roles[0], row['prompt'])
                    # conv.append_message(conv.roles[1], None)
                    # prompt = conv.get_prompt()
                    prompts.append(row['prompt'])

                    labels.append(True)
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not implemented in interactive mode.")
        

    if prompts == []:
        raise ValueError("No prompts found for alignment.")
    if saved_prompts is None:
        saved_prompts = prompts

    all_prompt_responses = []
    for i, (prompt, label) in enumerate(zip(prompts, labels)):
        if categories and (len(categories) == len(prompts)):
            print(f"Category: {categories[i]}", flush=True)
        _, responses, _ = aligner.generate_batch_and_response([prompt])
        response = responses[0]
        preds, probs = classify_texts([response], torch.device(cfg.device), classifier_tokenizer, classifier)
        print(f"Prompt {i}: {prompt}", flush=True)
        print(f"New response: {response}", flush=True)
        print(f"preds = {preds}, probs = {probs}", flush=True)

        all_prompt_responses.append(
            {
                "prompt": saved_prompts[i],
                "formatted_prompt": prompt,
                "aligned_response": response,
                "base_pred": preds[0],
                "base_probs": probs[0],
                "aligned_pred": preds[0],
                "aligned_probs": probs[0],
                "label": label,
            }
        )
    
    if args.no_save:
        print("Not saving results (--no_save).", flush=True)
        return

    save_prompt_responses_path = os.path.join(save_dir, "prompt_responses.json")
    print(f"Saving all prompt responses to {save_prompt_responses_path}", flush=True)

    with open(save_prompt_responses_path, "w") as f:
        json.dump(all_prompt_responses, f, indent=4)

if __name__ == "__main__":
    main()
