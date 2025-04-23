from typing import List
import torch
from transformers import PreTrainedTokenizer

class TripletCollator:
    """Преобразует batch из triplet-примеров в токены для (q, pos, negs...)."""

    def __init__(self, tokenizer: PreTrainedTokenizer, max_len: int, device: str):
        self.tok = tokenizer
        self.max_len = max_len
        self.dev = device

    def __call__(self, batch: List[dict]) -> dict:
        texts: List[str] = []
        for ex in batch:
            texts.append(f"search_query: {ex['query']}")
            texts.append(f"search_document: {ex['positive']}")
            texts.extend(f"search_document: {n}" for n in ex['negative'])
        enc = self.tok(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {k: v.to(self.dev) for k, v in enc.items()}
