from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from omegaconf import DictConfig
import os
import time
from tqdm import tqdm
from typing import Optional, Literal, List, Union, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from value_function_lib import value_function

BFLOAT16_MODELS = [
    "openai/gpt-oss-20b",
    "tiiuae/falcon-7b-instruct",
]

@dataclass
class AlignerType(Enum):
    BRT_LEAST_RESTRICTIVE = "brt_least_restrictive"
    BRT_BLEND = "brt_blend"


# ----------------------------
# ε-ball sampler (uniform volume for L2)
# ----------------------------
@dataclass
class NoiseConfig:
    radius_mode: Literal["uniform_radius", "uniform_volume"] = "uniform_radius"
    epsilon_search: float = 0.0
    epsilon_cap: float = 0.0
    K: int = 16
    norm: Literal["l2", "linf"] = "l2"
    include_base: bool = True

# All aligner-specific configs
@dataclass
class LeastRestrictiveReachableAlignerSpec:
    noise_cfg: NoiseConfig
    value_hidden_dims: List[int]
    value_threshold: float
    value_model_ckpt: str

@dataclass
class SmoothBlendReachableAlignerSpec:
    noise_cfg: NoiseConfig
    value_hidden_dims: List[int]
    value_coeff_gamma: float
    value_model_ckpt: str
    value_binary_search_max_iter: int = 2
    
@dataclass
class SAPAlignerSpec:
    cfg: DictConfig

@dataclass
class ReControlAlignerSpec:
    num_updates: int
    step_size: float
    value_model_ckpt: str
    value_hidden_dims: List[int]

@dataclass
class GradientAscentReachableAlignerSpec:
    num_updates: int
    step_size: float
    value_model_ckpt: str
    value_hidden_dims: List[int]
    value_threshold: float

# ----------------------------
# Aligner
# ----------------------------
@dataclass
class AlignConfig:
    # Shared Information.
    dataset_name: str
    split_name: str
    model_name: str
    layer_idx: int = 20
    max_new_tokens: int = 128
    seed: int = 0
    device: str = "cuda:0"
    bf16: bool = False
    num_shards: int = 1
    shard_idx: int = 0
    save_descriptor: str = ""
    sample_dataset: Optional[int] = None
    only_unsafe: bool = False
    calibration: bool = False
    use_prompt_engineering: bool = False

    # New: turn stochastic decoding on or off
    stochastic_decoding: bool = False

    # Aligner-Specific Information.
    aligner_spec: Union[LeastRestrictiveReachableAlignerSpec, SmoothBlendReachableAlignerSpec, SAPAlignerSpec, None] = None
    aligner_save_name: Optional[str] = None
    brt_type: str = None

# ----------------------------
# Model block locator
# ----------------------------
def get_block_list_and_name(causal_lm: torch.nn.Module) -> Tuple[torch.nn.ModuleList, str]:
    m = causal_lm
    candidates = [
        ("model.model.layers",        lambda m: m.model.model.layers),     # e.g., some LLaMA wrappers
        ("model.layers",              lambda m: m.model.layers),           # LLaMA/Mistral/Qwen style
        ("model.transformer.h",       lambda m: m.model.transformer.h),    # some Falcon builds
        ("transformer.h",             lambda m: m.transformer.h),          # Falcon/GPT-2 style (top-level)
        ("model.gpt_neox.layers",     lambda m: m.model.gpt_neox.layers),  # GPT-NeoX
        ("gpt_neox.layers",           lambda m: m.gpt_neox.layers),        # alternate NeoX
        ("model.decoder.layers",      lambda m: m.model.decoder.layers),   # some encoder-decoder or MPT variants
        ("backbone.model.layers",     lambda m: m.backbone.model.layers),  # occasional custom remotes
    ]
    for name, fn in candidates:
        try:
            layers = fn(m)
            if isinstance(layers, (torch.nn.ModuleList, list)) and len(layers) > 0:
                return layers, name
        except Exception:
            pass

    # Generic fallback: scan common parents/attrs
    parents = [("self", m),
               ("model", getattr(m, "model", None)),
               ("transformer", getattr(m, "transformer", None)),
               ("backbone", getattr(m, "backbone", None))]
    attrs = ["layers", "h", "blocks"]
    for p_name, parent in parents:
        if parent is None: 
            continue
        for attr in attrs:
            if hasattr(parent, attr):
                layers = getattr(parent, attr)
                if isinstance(layers, (torch.nn.ModuleList, list)) and len(layers) > 0:
                    return layers, f"{p_name}.{attr}"

    raise RuntimeError("Unsupported architecture: cannot locate transformer block list.")

# ----------------------------
# Value model loader
# ----------------------------
def load_value_function(path_or_tag: Optional[str], input_dim: int, hidden_dims: List[int], device) -> nn.Module:
    model = value_function.ValueFunction(input_dim=input_dim, hidden_dims=hidden_dims)
    model = model.to(device=device, dtype=torch.bfloat16)  # Always use float16 for the value model
    model.eval()
    if path_or_tag:
        sd = torch.load(path_or_tag, map_location="cpu")
        # Convert all parameters to float16 when loading
        sd = {k: v.to(dtype=torch.bfloat16) for k, v in sd.items()}
        model.load_state_dict(sd)
    return model

class BaseAligner(ABC):
    def __init__(self, cfg: AlignConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = torch.float32 if cfg.model_name not in BFLOAT16_MODELS else torch.bfloat16

        set_seed(cfg.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        self.lm = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=self.dtype,
            output_hidden_states=False,
            trust_remote_code=True,
        ).to(self.device).eval()

        self.blocks, _ = get_block_list_and_name(self.lm)
        try:
            self.hidden_size = self.lm.config.hidden_size
        except Exception:
            with torch.no_grad():
                tmp = self.lm(input_ids=torch.tensor([[self.tokenizer.eos_token_id]], device=self.device))
            self.hidden_size = tmp.hidden_states[-1].shape[-1]

    def reset(self):
        pass

    def _target_block_for_hidden_state_index(self) -> torch.nn.Module:
        """
        Match cfg.layer_idx to HuggingFace output_hidden_states indexing.

        output_hidden_states[0] is the token-embedding output, and
        output_hidden_states[i] for i > 0 is the output of transformer block i-1.
        """
        if self.cfg.layer_idx <= 0:
            raise ValueError(
                "cfg.layer_idx=0 refers to embedding outputs, which cannot be "
                "intervened on with a transformer-block forward hook."
            )
        block_idx = self.cfg.layer_idx - 1
        if block_idx >= len(self.blocks):
            raise ValueError(
                f"cfg.layer_idx={self.cfg.layer_idx} maps to block {block_idx}, "
                f"but this model only has {len(self.blocks)} transformer blocks."
            )
        return self.blocks[block_idx]

    def _past_len(self, past) -> int:
        return past[0][0].shape[-2] if (past and len(past) > 0) else 0

    def _decode_like_dataset(self, prompt: str, response_so_far_ids: torch.Tensor) -> str:
        # 1) decode prompt + generated_so_far, 2) slice by char len of prompt, 3) strip
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(self.device)
        full_ids = torch.cat([prompt_ids[0], response_so_far_ids[0]], dim=0).unsqueeze(0)
        decoded_full = self.tokenizer.decode(full_ids[0], skip_special_tokens=True)
        return decoded_full[len(prompt):].strip()
    
    @abstractmethod
    def _hook_aligner(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.utils.hooks.RemovableHandle:
        raise NotImplementedError("Subclasses must implement _hook_aligner.")
    
    @abstractmethod
    def _hook_aligner_batched(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.utils.hooks.RemovableHandle:
        raise NotImplementedError("Subclasses must implement _hook_aligner.")

    def generate(self, prompt: str) -> str:
        """
        SINGLE PATH:
        - Generation stream: from RAW prompt (no '\n' injected).
        - At each step, build a PROBE stream using (prompt + '\n' + response_so_far)
          to get h_last consistent with training.
        """
        tok = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=512
        ).to(self.device)
        input_ids = tok.input_ids  # raw prompt tokens
        T = input_ids.shape[1]

        # Prime generation cache on all but last prompt token
        if T > 1:
            prime = self.lm(
                input_ids=input_ids[:, :-1],
                attention_mask=tok.attention_mask[:, :-1],
                use_cache=True,
                return_dict=True,
            )
            past_gen = prime.past_key_values
        else:
            past_gen = None

        sequence = input_ids.clone()           # full generated sequence (prompt + tokens)
        running_mask = tok.attention_mask[:, :-1].clone()
        curr_input = input_ids[:, -1:]         # last prompt token (step 0 conditioning)
        response_so_far = torch.empty((1, 0), dtype=torch.long, device=self.device)  # for PROBE only
        value_preds = []                       # record value model preds at each step

        for i in range(self.cfg.max_new_tokens):
            # Clear cache periodically
            if i > 0 and i % 10 == 0:
                torch.cuda.empty_cache()

            # --- Generation step with Δ hook
            running_mask = torch.cat([running_mask, torch.ones_like(curr_input)], dim=1)
            pl = self._past_len(past_gen)
            if self.cfg.model_name in ["tiiuae/falcon-7b-instruct"]:
                # --- PROBE: get h_last under training-consistent context
                with torch.no_grad():
                    kwargs = {
                        "past_key_values": past_gen,
                        "use_cache": True,
                        "return_dict": True,
                    }
                    handle = self._hook_aligner(curr_input, running_mask, **kwargs)
                    out_step = self.lm(
                        input_ids=curr_input,
                        attention_mask=running_mask,
                        past_key_values=past_gen,
                        use_cache=True,
                        return_dict=True,
                    )
                    handle.remove()
            else:
                cache_position = torch.arange(pl, pl + 1, device=self.device, dtype=torch.long)
                position_ids   = cache_position.unsqueeze(0)

                # --- PROBE: get h_last under training-consistent context
                with torch.no_grad():
                    kwargs = {
                        "past_key_values": past_gen,
                        "use_cache": True,
                        "return_dict": True,
                    }
                    handle = self._hook_aligner(curr_input, running_mask, **kwargs)
                    out_step = self.lm(
                        input_ids=curr_input,
                        attention_mask=running_mask,
                        past_key_values=past_gen,
                        use_cache=True,
                        cache_position=cache_position,
                        position_ids=position_ids,
                        return_dict=True,
                    )
                    handle.remove()
                
            logits_last = out_step.logits[:, -1, :]  # [1, V]

            if self.cfg.stochastic_decoding:
                # top_p = 0.95 sampling
                probs = torch.softmax(logits_last.float(), dim=-1)  # [1, V]
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                cumprobs = torch.cumsum(sorted_probs, dim=-1)

                top_p = 0.95
                mask = cumprobs > top_p
                mask[..., 0] = False               # always keep the most likely token
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)

                # Renormalize
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                # Sample from truncated distribution
                next_rel = torch.multinomial(sorted_probs, num_samples=1)   # [1, 1]
                next_id = sorted_indices.gather(-1, next_rel)               # [1, 1]
            else:
                # Original greedy behavior
                next_id = logits_last.argmax(dim=-1, keepdim=True)          # [1, 1]

            sequence = torch.cat([sequence, next_id], dim=1)

            # update states
            past_gen = out_step.past_key_values
            curr_input = next_id
            response_so_far = torch.cat([response_so_far, next_id], dim=1)

            if next_id.item() == self.tokenizer.eos_token_id:
                break

        resp_ids = sequence[0, T:]               # tokens after the prompt
        response_text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
        return response_text, value_preds

    def generate_batch_and_response(self, prompts: List[str]) -> Tuple[List[str], List[str], List[List[float]]]:
        assert len(prompts) > 0
        self.reset()
        
        tok = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,       # left padding already configured
            truncation=True,
            max_length=512
        ).to(self.device)

        input_ids = tok.input_ids         # [B, T]
        attn      = tok.attention_mask    # [B, T]
        B, T = input_ids.shape

        # Prime KV cache with TRUE mask
        with torch.no_grad():
            if T > 1:
                prime = self.lm(
                    input_ids=input_ids[:, :-1],
                    attention_mask=attn[:, :-1],
                    use_cache=True,
                    return_dict=True,
                )
                past_gen = prime.past_key_values
                del prime
            else:
                past_gen = None

        sequence     = input_ids.clone()           # [B, T]
        running_mask = attn[:, :-1].clone()        # [B, T-1]
        curr_input   = input_ids[:, -1:]           # [B, 1]
        eos_id       = self.tokenizer.eos_token_id
        finished     = torch.zeros(B, dtype=torch.bool, device=self.device)

        # Per-sample scalar trace (e.g., value model preds each step)
        value_preds_per_sample: List[List[float]] = [[] for _ in range(B)]

        for _ in range(self.cfg.max_new_tokens):
            t_1 = time.perf_counter()

            # Extend mask by one step
            running_mask = torch.cat(
                [running_mask, torch.ones((B, 1), dtype=attn.dtype, device=self.device)],
                dim=1
            )

            # Keep finished rows shape-aligned
            if eos_id is not None:
                curr_input = torch.where(
                    finished.view(-1, 1),
                    torch.full_like(curr_input, eos_id),
                    curr_input
                )

            # Choose correct aligner hook
            hook_fn = (
                self._hook_aligner_batched
                if hasattr(self, "_hook_aligner_batched") and B > 1
                else self._hook_aligner
            )
            align_handle = hook_fn(
                curr_input,
                running_mask,
                past_key_values=past_gen,
                use_cache=True,
                return_dict=True,
            )

            # Run one generation step with no autograd
            with torch.no_grad():
                out_step = self.lm(
                    input_ids=curr_input,
                    attention_mask=running_mask,
                    past_key_values=past_gen,
                    use_cache=True,
                    return_dict=True,
                )

            align_handle.remove()

            # Greedy next token
            logits_last = out_step.logits[:, -1, :]  # [B, V]

            if self.cfg.stochastic_decoding:
                probs = torch.softmax(logits_last.float(), dim=-1)   # [B, V]
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                cumprobs = torch.cumsum(sorted_probs, dim=-1)

                top_p = 0.95
                mask = cumprobs > top_p
                mask[..., 0] = False
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)

                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_rel = torch.multinomial(sorted_probs, num_samples=1)   # [B, 1]
                next_id = sorted_indices.gather(-1, next_rel)               # [B, 1]
            else:
                next_id = logits_last.argmax(dim=-1, keepdim=True)          # [B, 1]

            if eos_id is not None:
                next_id = torch.where(
                    finished.view(-1, 1),
                    torch.full_like(next_id, eos_id),
                    next_id,
                )

            sequence  = torch.cat([sequence, next_id], dim=1)
            past_gen  = out_step.past_key_values
            curr_input = next_id

            # Track finished
            if eos_id is not None:
                finished |= (next_id.squeeze(-1) == eos_id)
                if finished.all():
                    break

            # Hygiene
            del out_step, next_id
            if sequence.size(1) % 16 == 0:
                torch.cuda.empty_cache()
            t_2 = time.perf_counter()

        # Decode per row after the prompt
        responses: List[str] = []
        T0 = input_ids.size(1)  # padded prompt width for this micro-batch
        for b in range(B):
            resp_ids_b = sequence[b, T0:]  # first generated token is at column T0
            responses.append(self.tokenizer.decode(resp_ids_b, skip_special_tokens=True).strip())

        return prompts, responses, value_preds_per_sample
