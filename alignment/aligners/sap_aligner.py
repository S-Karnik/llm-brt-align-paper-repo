import time
import torch

try:
    from safety_polytope.evaluation import mmlu
except ImportError:
    mmlu = None

from alignment.aligners.base_aligner import (
    AlignConfig,
    BaseAligner,
    SAPAlignerSpec
)

class SAPAligner(BaseAligner):
    def __init__(self, cfg: AlignConfig):
        super().__init__(cfg)
        assert isinstance(cfg.aligner_spec, SAPAlignerSpec)
        if mmlu is None:
            raise ImportError(
                "SaP alignment requires the external safety_polytope package. "
                "Install the SaP/Safety Polytope baseline separately to run this aligner."
            )

        self._safe_rep_model, self._tokenizer = mmlu.load_safe_rep_model(cfg.aligner_spec.cfg)
        self._safe_rep_model.to(self.device)
        self._safe_rep_model.to(self.dtype)

        self.steer_first_n_tokens = cfg.aligner_spec.cfg.steer_first_n_tokens
        self.tokens_steered = 0

    def _hook_aligner(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.utils.hooks.RemovableHandle:
        """
        Invokes the SaP hook for aligning the LLM.
        """
        target_block = self._target_block_for_hidden_state_index()
        start_time = time.time()
        
        def hook(_m, _inp, out):
            with torch.cuda.amp.autocast(enabled=False):
                is_tuple = isinstance(out, tuple)
                y = out[0] if is_tuple else out
                y2 = y.clone()
                
                if self.tokens_steered < self.steer_first_n_tokens:
                    features = self._safe_rep_model.feature_extractor(y2[:, -1, :])
                    safe_mask = ~self._safe_rep_model.check_constraint(features)

                    if safe_mask.any():
                        y2[:, -1, :] = self._safe_rep_model._apply_optimization(
                            y2[:, -1, :], safe_mask, start_time
                        )

                    self.tokens_steered += 1
                    del features
                    del safe_mask
                            
            return (y2,) + out[1:] if is_tuple else y2
        
        return target_block.register_forward_hook(hook)

    def _hook_aligner_batched(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> torch.utils.hooks.RemovableHandle:
        """
        Batched gradient ascent on the value model to find better hidden states.
        Each batch element is optimized independently.
        """
        target_block = self._target_block_for_hidden_state_index()
        start_time = time.time()

        def hook(_m, _inp, out):
            with torch.cuda.amp.autocast(enabled=False):
                is_tuple = isinstance(out, tuple)
                y = out[0] if is_tuple else out  # [B, T, D]
                y2 = y.clone()

                if self.tokens_steered < self.steer_first_n_tokens:
                    features = self._safe_rep_model.feature_extractor(y2[:, -1, :])
                    safe_mask = ~self._safe_rep_model.check_constraint(features)

                    if safe_mask.any():
                        for i in range(y2.size(0)):
                            if safe_mask[i]:
                                y2[i, -1, :] = self._safe_rep_model._apply_optimization(
                                    y2[i, -1, :].unsqueeze(0), safe_mask[i].unsqueeze(0), start_time
                                )

                    self.tokens_steered += 1
                    del features
                    del safe_mask
            return (y2,) + out[1:] if is_tuple else y2

        return target_block.register_forward_hook(hook)
    
    def reset(self):
        self.tokens_steered = 0
