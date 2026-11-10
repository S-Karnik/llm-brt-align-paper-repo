from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional

class CosineSimilarity:
    def __init__(
        self,
        model_name: str = "princeton-nlp/sup-simcse-roberta-large",
        device: Optional[str] = None,        # e.g. "cuda", "cuda:0", "cpu", "mps"
        dtype: str = "fp32",                 # "fp32", "bf16", "fp16"
        max_length: int = 512
    ):
        # Choose device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = torch.device(device)

        # Dtype
        _dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
        self._dtype = _dtype_map[dtype]

        # Load model/tokenizer once
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self._device)
        self._model.eval()

        self._max_length = max_length

        # Simple in-memory cache: text -> np.ndarray normalized embedding
        self._cache: Dict[str, np.ndarray] = {}
    
    def reset_cache(self):
        self._cache = {}

    @torch.no_grad()
    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts, return L2-normalized embeddings as float32 numpy array."""
        if len(texts) == 0:
            return np.empty((0, 1024), dtype=np.float32)  # placeholder

        inputs = self._tokenizer(
            texts, padding=True, truncation=True, max_length=self._max_length, return_tensors="pt"
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.autocast(device_type=self._device.type, enabled=False):
            outputs = self._model(**inputs, output_hidden_states=True, return_dict=True)
            pooled = outputs.pooler_output  # SimCSE uses pooler_output for sentence vectors
            if pooled.dtype is not self._dtype:
                pooled = pooled.to(self._dtype)
            pooled = F.normalize(pooled, p=2, dim=1)

        return pooled.to(torch.float32).cpu().numpy()

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get (or cache) a single sentence embedding."""
        text = text or ""
        emb = self._cache.get(text)
        if emb is None:
            emb = self._encode([text])[0]
            self._cache[text] = emb
        return emb

    def sentence_similarity(self, sent1: str, sent2: str) -> float:
        emb1 = self._get_embedding(sent1)
        emb2 = self._get_embedding(sent2)
        return float(np.dot(emb1, emb2))  # cosine similarity

    def batch_sentence_similarity(
        self,
        sents1: List[str],
        sents2: List[str],
        batch_size: int = 32
    ) -> List[float]:
        """Compute cosine similarities for pairs of sentences in two lists (batched)."""
        assert len(sents1) == len(sents2), "Lists must have the same length"
        sims: List[float] = []

        # Process in chunks
        for start in range(0, len(sents1), batch_size):
            end = start + batch_size
            chunk1 = sents1[start:end]
            chunk2 = sents2[start:end]

            # Collect uncached sentences
            uncached1 = [s for s in chunk1 if s not in self._cache]
            uncached2 = [s for s in chunk2 if s not in self._cache]

            # Encode uncached and update cache
            if uncached1:
                embs1 = self._encode(uncached1)
                for s, e in zip(uncached1, embs1):
                    self._cache[s] = e
            if uncached2:
                embs2 = self._encode(uncached2)
                for s, e in zip(uncached2, embs2):
                    self._cache[s] = e

            # Gather embeddings from cache
            arr1 = np.stack([self._cache[s] for s in chunk1])
            arr2 = np.stack([self._cache[s] for s in chunk2])

            # Cosine similarity is dot product (since embeddings are normalized)
            sims.extend(np.sum(arr1 * arr2, axis=1).tolist())

        return sims


if __name__ == "__main__":
    cos_sim = CosineSimilarity()
    list1 = ["This is a test sentence.", "Another one here."]
    list2 = ["This is a test sentence!", "Completely different sentence."]
    sims = cos_sim.batch_sentence_similarity(list1, list2)
    for a, b, s in zip(list1, list2, sims):
        print(f"'{a}' vs '{b}' → cosine similarity = {s:.4f}")
