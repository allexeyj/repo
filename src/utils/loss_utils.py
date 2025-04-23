import torch
import torch.nn.functional as F

def info_nce_loss(anchors: torch.Tensor,
                  positives: torch.Tensor,
                  batch_negs: torch.Tensor,
                  queue_embs: torch.Tensor,
                  temperature: float) -> torch.Tensor:
    """
    anchors    – [B, H]
    positives  – [B, H]
    batch_negs – [N_b, H]  (может быть пустым)
    queue_embs – [N_q, H]  (может быть пустым)
    """
    # --- L2‑нормализация ---
    anchors = F.normalize(anchors, dim=-1)
    positives = F.normalize(positives, dim=-1)

    refs = [positives]
    if batch_negs.numel():
        refs.append(F.normalize(batch_negs, dim=-1))
    if queue_embs.numel():
        refs.append(F.normalize(queue_embs, dim=-1))
    refs = torch.cat(refs, dim=0)  # [P | N_b | N_q]

    # --- логиты ---
    logits = torch.matmul(anchors, refs.T) / temperature  # [B, P+N]
    targets = torch.arange(anchors.size(0), device=anchors.device)

    # cross‑entropy сама сделает softmax + log + mean
    loss = F.cross_entropy(logits, targets)
    return loss