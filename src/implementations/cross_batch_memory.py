import torch
from collections import deque


class CrossBatchMemory:
    '''
    очередь эмбеддингов предыдущих негативов для более стабильного InfoNCE
    нужен большой батч, а большой батч наши видеокарты не тянут, очередь "симулирует" большой батч.
    '''

    def __init__(self, max_size: int, dim: int, device: str):
        self.max_size = max_size
        self.device = device
        self.embeddings = deque()
        self.dim = dim

    @torch.no_grad()
    def enqueue(self, embs: torch.Tensor):
        for e in embs.detach():
            if len(self.embeddings) == self.max_size:
                self.embeddings.popleft()
            self.embeddings.append(e)

    def get(self) -> torch.Tensor:
        if len(self.embeddings) == 0:
            return torch.empty(0, self.dim, device=self.device)
        return torch.stack(tuple(self.embeddings))
