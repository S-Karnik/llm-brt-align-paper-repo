from dataclasses import dataclass
from typing import Optional, Tuple, Literal, List, Union

import numpy as np
import time
import torch

from alignment.aligners.base_aligner import (
    AlignConfig,
    BaseAligner,
    NoiseConfig,
    GradientAscentReachableAlignerSpec,
    LeastRestrictiveReachableAlignerSpec,
    SmoothBlendReachableAlignerSpec,
    load_value_function,
)

BFLOAT16_MODELS = [
    "openai/gpt-oss-20b",
    "tiiuae/falcon-7b-instruct",
]

class EmbeddingNoiseSampler:
    """
    Sampling around `base` with a wide search radius.

    - L2 sampling uses directions ~ uniform on sphere and radius ~ Uniform[0, eps_search]
    - L∞ sampling uses a uniform hypercube [-eps_search, eps_search]^D
    - epsilon_cap is recorded in configs/results but is not used to reject samples
    """

    def __init__(self, cfg: NoiseConfig, device: torch.device, dtype: torch.dtype):
        """
        radius_mode: "uniform_radius" (default) or "uniform_volume"
            - uniform_radius: R ~ Uniform[0, eps_search]
            - uniform_volume: R = eps_search * U^(1/D)  (concentrates near boundary in high D)
        """
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

    @staticmethod
    def _valid_l2_origin(x: torch.Tensor, r: float) -> torch.Tensor:
        # x: [..., D]
        if r is None or r <= 0:
            return torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
        return x.norm(p=2, dim=-1) <= r

    @staticmethod
    def _valid_linf_origin(x: torch.Tensor, r: float) -> torch.Tensor:
        # x: [..., D]
        if r is None or r <= 0:
            return torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
        return x.abs().amax(dim=-1) <= r

    def valid_mask(self, x: torch.Tensor, abs_cap_kind: str = "l2") -> torch.Tensor:
        return torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)

    @torch.no_grad()
    def sample(
        self,
        base: torch.Tensor,                  # [B, D]
        abs_cap_kind: str = "l2",            # 'l2' or 'linf' (absolute cap type)
        return_valid_mask: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns K candidates of shape [K, B, D].

        - If eps_search==0, returns K copies of base.
        - The returned validity mask is all True; candidates are not rejected by
          epsilon_cap.
        """
        B, D = base.shape
        K = self.cfg.K
        norm = self.cfg.norm
        eps = self.cfg.epsilon_search

        k0 = 1 if self.cfg.include_base else 0
        out = torch.empty((K, B, D), dtype=self.dtype, device=self.device)

        # Fill the base candidate if requested.
        if k0:
            out[0] = base.to(self.dtype)

        # Early exit if no exploration
        if eps is None or eps <= 0 or K - k0 <= 0:
            if K - k0 > 0:
                out[k0:] = out[0].unsqueeze(0).expand(K - k0, -1, -1) if k0 else base.unsqueeze(0).expand(K, -1, -1)
            return (out, self.valid_mask(out, abs_cap_kind)) if return_valid_mask else out

        # ----- Sampling -----
        if norm == "l2":
            # Directions ~ N(0,I) normalized
            dirs = torch.randn((K - k0, B, D), dtype=torch.float32, device=self.device)
            dirs = dirs / dirs.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

            if self.cfg.radius_mode == "uniform_radius":
                radii = eps * torch.rand((K - k0, B, 1), dtype=torch.float32, device=self.device)
            elif self.cfg.radius_mode == "uniform_volume":
                # R = eps * U^(1/D)
                radii = eps * torch.rand((K - k0, B, 1), dtype=torch.float32, device=self.device).pow(1.0 / D)
            else:
                raise ValueError(f"Unknown radius_mode: {self.cfg.radius_mode}")

            deltas = (dirs * radii).to(self.dtype)                # [K-k0,B,D]
            cands = base.unsqueeze(0).to(self.dtype) + deltas     # [K-k0,B,D]

            out[k0:] = cands

        elif norm == "linf":
            # Uniform hypercube
            deltas = (torch.rand((K - k0, B, D), dtype=torch.float32, device=self.device) * 2.0 - 1.0) * float(eps)
            cands = base.unsqueeze(0).to(self.dtype) + deltas.to(self.dtype)

            out[k0:] = cands

        else:
            raise ValueError(f"Unknown norm: {norm}")

        return (out, self.valid_mask(out, abs_cap_kind)) if return_valid_mask else out


class LeastRestrictiveReachableAligner(BaseAligner):
    def __init__(self, cfg: AlignConfig):
        super().__init__(cfg)
        assert isinstance(cfg.aligner_spec, LeastRestrictiveReachableAlignerSpec)
        
        self._value_model = load_value_function(cfg.aligner_spec.value_model_ckpt, self.hidden_size, cfg.aligner_spec.value_hidden_dims, self.device)
        self._noise = EmbeddingNoiseSampler(
            cfg.aligner_spec.noise_cfg,
            device=self.device,
            dtype=torch.float32,
        )

    def _hook_aligner(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.utils.hooks.RemovableHandle:
        """
        Adds 'delta' to the last token's hidden at cfg.layer_idx in the GENERATION stream.
        """
        target_block = self._target_block_for_hidden_state_index()

        def hook(_m, _inp, out):
            if self.cfg.aligner_spec.noise_cfg.epsilon_search <= 0:
                return out

            is_tuple = isinstance(out, tuple)
            y = out[0] if is_tuple else out
            y2 = y.clone()
            default_val = self._value_model(y2[:, -1, :].to(torch.bfloat16)).item()
                        
            # If the value is above the threshold, do nothing.
            if default_val > self.cfg.aligner_spec.value_threshold:
                return out
            
            # Otherwise, find the best candidate.
            cands, valid = self._noise.sample(y2[:, -1, :], return_valid_mask=True)
            cands = cands.to(torch.bfloat16)
            vals = self._value_model(cands[:, 0, :].to(torch.bfloat16)).reshape(-1)
            vals = vals.masked_fill(~valid[:, 0], float("-inf"))
            if not torch.any(valid[:, 0]):
                return out
            j = torch.argmax(vals).item()
            y2[:, -1, :] = cands[j, 0, :]
            return (y2,) + out[1:] if is_tuple else y2

        return target_block.register_forward_hook(hook)
    
    def _hook_aligner_batched(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.utils.hooks.RemovableHandle:
        """
        Batched least-restrictive hook:
        - Evaluate value per row at the last token.
        - If a row is below threshold, search over K candidates around its last hidden and
        pick the best-scoring one.
        - Leave rows above threshold unchanged.
        """
        target_block = self._target_block_for_hidden_state_index()

        def hook(_m, _inp, out):
            if self.cfg.aligner_spec.noise_cfg.epsilon_search <= 0:
                return out

            t_1 = time.perf_counter()

            # out is either Tensor [B, T, D] or tuple with hidden states first
            is_tuple = isinstance(out, tuple)
            y = out[0] if is_tuple else out                  # [B, T, D]
            B, T, D = y.shape

            # Work on a view; we'll clone only if we actually modify something
            y_last = y[:, -1, :]                             # [B, D]

            # 1) Default values per row (value model expects bf16)
            with torch.no_grad():
                default_vals = self._value_model(y_last.to(torch.bfloat16)).reshape(-1)  # [B]
            thresh = self.cfg.aligner_spec.value_threshold
            need_mask = default_vals <= thresh                                                 # [B]

            # If nothing needs intervention, return original out exactly
            if not torch.any(need_mask):
                return out

            # 2) Sample K candidates around each row's last hidden: [K, B, D]
            #    The sampler returns float32 (as configured); keep it to score, then cast back.
            cands, valid = self._noise.sample(y_last.to(torch.float32), return_valid_mask=True) # [K, B, D], [K, B]

            # 3) Score all candidates in a single pass: reshape to [K*B, D] -> values -> [K, B]
            K = cands.shape[0]
            cands_flat = cands.reshape(K * B, D)
            with torch.no_grad():
                vals = self._value_model(cands_flat.to(torch.bfloat16)).reshape(K, B)         # [K, B]
            vals = vals.masked_fill(~valid, float("-inf"))
            has_valid = torch.any(valid, dim=0)                                                # [B]

            # 4) Best candidate per row
            best_k = torch.argmax(vals, dim=0)                                                 # [B]
            arange_B = torch.arange(B, device=cands.device)
            best_per_row = cands[best_k, arange_B, :]                                          # [B, D]

            # 5) Write back only rows that need it (clone lazily to preserve identity otherwise)
            y2 = y.clone()
            write_mask = need_mask & has_valid
            y2[write_mask, -1, :] = best_per_row[write_mask].to(y.dtype)

            t_2 = time.perf_counter()

            print(f"[LeastRestrictiveReachableAligner] Batched hook time: {t_2 - t_1:.4f} seconds")

            return (y2,) + out[1:] if is_tuple else y2

        return target_block.register_forward_hook(hook)


class SmoothBlendReachableAligner(BaseAligner):
    def __init__(self, cfg: AlignConfig):
        super().__init__(cfg)
        assert isinstance(cfg.aligner_spec, SmoothBlendReachableAlignerSpec)
        
        self._value_model = load_value_function(cfg.aligner_spec.value_model_ckpt, self.hidden_size, cfg.aligner_spec.value_hidden_dims, self.device)
        self._noise = EmbeddingNoiseSampler(
            cfg.aligner_spec.noise_cfg,
            device=self.device,
            dtype=torch.float32,
        )

    def _hook_aligner(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.utils.hooks.RemovableHandle:
        """
        Uses smooth blending to find the best candidate.
        
        - The basic idea is to search for \eps such that V(x + \eps) >= e^(-\gamma) * V(x).
        - We want to minimize ||\eps||_2, so we use binary search.
        """
        target_block = self._target_block_for_hidden_state_index()

        def hook(_m, _inp, out):
            # Disable autocast for this hook to prevent dtype issues
            with torch.cuda.amp.autocast(enabled=False):
                is_tuple = isinstance(out, tuple)
                y = out[0] if is_tuple else out
                y2 = y.clone()
                                
                # Convert to float32 for the value model
                y2_float32 = y2.to(torch.float32)
                default_val = self._value_model(y2_float32[:, -1, :].to(torch.bfloat16)).item()
                
                # Sample candidates and ensure they're in float32
                cands, valid = self._noise.sample(y2_float32[:, -1, :], return_valid_mask=True)
                vals = self._value_model(cands[:, 0, :].to(torch.bfloat16)).reshape(-1)
                vals = vals.masked_fill(~valid[:, 0], float("-inf"))
                if not torch.any(valid[:, 0]):
                    return out
                
                j = torch.argmax(vals).item()
                max_val = vals[j].item()
                last_aligned_y2 = cands[j, 0, :]
                                                
                if max_val < np.exp(-self.cfg.aligner_spec.value_coeff_gamma) * default_val:
                    # If the value is below the threshold, return the closest one.
                    y2[:, -1, :] = last_aligned_y2.to(y.dtype)
                else:
                    # Otherwise, find the best candidate using binary search
                    new_aligned_y2 = last_aligned_y2
                    last_valid_aligned_y2 = last_aligned_y2
                    binary_search_count = 0
                    while max_val > np.exp(-self.cfg.aligner_spec.value_coeff_gamma) * default_val and binary_search_count < self.cfg.aligner_spec.value_binary_search_max_iter:
                        if self._noise.valid_mask(new_aligned_y2.unsqueeze(0))[0]:
                            last_valid_aligned_y2 = new_aligned_y2
                        new_aligned_y2 = (new_aligned_y2 + y2_float32[:, -1, :]) / 2
                        max_val = self._value_model(new_aligned_y2.unsqueeze(0).to(torch.bfloat16)).item()
                        binary_search_count += 1
                    if self._noise.valid_mask(new_aligned_y2.unsqueeze(0))[0]:
                        y2[:, -1, :] = new_aligned_y2.to(y.dtype)
                    else:
                        y2[:, -1, :] = last_valid_aligned_y2.to(y.dtype)
                
                return (y2,) + out[1:] if is_tuple else y2

        return target_block.register_forward_hook(hook)

class GradientAscentReachableAligner(BaseAligner):
    def __init__(self, cfg: AlignConfig):
        super().__init__(cfg)
        assert isinstance(cfg.aligner_spec, GradientAscentReachableAlignerSpec)
        
        self._value_model = load_value_function(cfg.aligner_spec.value_model_ckpt, self.hidden_size, cfg.aligner_spec.value_hidden_dims, self.device)

    def _hook_aligner(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.utils.hooks.RemovableHandle:
        """
        Invokes gradient ascent on the value model to find a better hidden state.
        """
        target_block = self._target_block_for_hidden_state_index()

        def hook(_m, _inp, out):
            is_tuple = isinstance(out, tuple)
            y = out[0] if is_tuple else out
            y2 = y.clone()
            
            # Otherwise, perform gradient ascent to find a better candidate.
            hs0 = y2[:, -1, :].detach().to(torch.bfloat16)                    # (B, D), leaf (detached)
            hidden_state = torch.nn.Parameter(hs0, requires_grad=True)        # make it optimizable (leaf)

            # Small optimizer; Adam is robust, but SGD works too
            opt = torch.optim.SGD([hidden_state], lr=self.cfg.aligner_spec.step_size)

            # Make sure value model is on the same device/dtype path
            # (load_value_function should already have moved it to self.device in fp32)
            with torch.enable_grad():
                for t in range(self.cfg.aligner_spec.num_updates):
                    opt.zero_grad(set_to_none=True)
                    output = self._value_model(hidden_state)                  # shape (B,) or (B,1)
                    loss = -output.reshape(-1).sum()                          # maximize value => minimize -sum
                    loss.backward()

                    # Optional diagnostics (uncomment if debugging):
                    # print(f"[iter {t}] grad mean={hidden_state.grad.abs().mean().item():.3e}")
                    opt.step()
        
            # Write the optimized fp32 state back into y2, casting to original dtype
            y2[:, -1, :] = hidden_state.detach().to(y2.dtype)
            return (y2, ) + out[1:] if is_tuple else y2
        return target_block.register_forward_hook(hook)

    def _hook_aligner_batched(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> torch.utils.hooks.RemovableHandle:
        """
        Batched gradient ascent on the value model to find better hidden states.
        Each batch element is optimized independently.
        """
        target_block = self._target_block_for_hidden_state_index()

        def hook(_m, _inp, out):
            is_tuple = isinstance(out, tuple)
            y = out[0] if is_tuple else out  # [B, T, D]
            y2 = y.clone()
            B, D = y2.shape[0], y2.shape[-1]

            # Initialize hidden states for optimization (one per example)
            hs0 = y2[:, -1, :].detach()  # [B, D]
            hidden_state = torch.nn.Parameter(hs0.clone(), requires_grad=True)

            opt = torch.optim.SGD([hidden_state], lr=self.cfg.aligner_spec.step_size)
            
            with torch.enable_grad():
                for _ in range(self.cfg.aligner_spec.num_updates):
                    opt.zero_grad(set_to_none=True)

                    # Value predictions per example: [B]
                    output = self._value_model(hidden_state.to(torch.bfloat16))  

                    # Maximize independently → sum of losses but grads separate per row
                    loss = -output.reshape(-1).sum()
                    loss.backward()
                    opt.step()

            # Write optimized states back into y2
            y2[:, -1, :] = hidden_state.detach().to(y2.dtype)
            return (y2,) + out[1:] if is_tuple else y2

        return target_block.register_forward_hook(hook)
