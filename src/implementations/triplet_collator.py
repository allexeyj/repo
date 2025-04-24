from typing import List
from transformers import PreTrainedTokenizer

class TripletCollator:
    """Преобразует batch из triplet-примеров в токены для (q, pos, negs...)."""

    def __init__(self, tokenizer: PreTrainedTokenizer, max_len: int):
        self.tok = tokenizer
        self.max_len = max_len

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
        # Отдаём на CPU, Accelerator перенесёт на GPU при prepare()
        return enc
