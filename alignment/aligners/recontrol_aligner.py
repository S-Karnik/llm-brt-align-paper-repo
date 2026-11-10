
import torch

from alignment.aligners.base_aligner import (
    AlignConfig,
    BaseAligner,
    ReControlAlignerSpec,
    load_value_function
)

class ReControlAligner(BaseAligner):
    def __init__(self, cfg: AlignConfig):
        super().__init__(cfg)
        assert isinstance(cfg.aligner_spec, ReControlAlignerSpec)

        self._value_model = load_value_function(cfg.aligner_spec.value_model_ckpt, self.hidden_size, cfg.aligner_spec.value_hidden_dims, self.device)

    def _hook_aligner(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.utils.hooks.RemovableHandle:
        """
        Invokes the SaP hook for aligning the LLM.
        """
        target_block = self._target_block_for_hidden_state_index()
        
        def hook(_m, _inp, out):
            # out can be tuple; we always work on the hidden state tensor y
            is_tuple = isinstance(out, tuple)
            y = out[0] if is_tuple else out

            # Work on a detached copy of the whole layer output
            y2 = y.detach().clone()

            # Take the last-token hidden state for each batch item and make it a *leaf* Parameter in fp32
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
                    opt.step()

            # Write the optimized fp32 state back into y2, casting to original dtype
            y2[:, -1, :] = hidden_state.detach().to(y2.dtype)

            # Return tuple structure unchanged
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
