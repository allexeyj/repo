import torch
from tqdm.auto import tqdm
from torch.nn.utils import clip_grad_norm_
from src.utils.loss_utils import info_nce_loss
import wandb
from torch import amp


def train_epoch(model, train_dl, optim, scheduler, memory, scaler, cfg, epoch_idx):
    model.train()
    total_loss = 0
    global_step = (epoch_idx - 1) * len(train_dl)
    for step, batch in tqdm(enumerate(train_dl), desc=f"Epoch {epoch_idx}/{cfg.training.epochs} [train]",
                            total=len(train_dl)):
        optim.zero_grad()
        with amp.autocast(enabled=(cfg.device == "cuda")):
            embs = model(**batch).last_hidden_state[:, 0, :]  # CLS

        # ---------- разбор батча ----------
        stride = 2 + cfg.batch.num_hard_negs
        anchors = embs[0::stride]
        positives = embs[1::stride]

        mask = torch.ones(embs.size(0), dtype=torch.bool, device=embs.device)
        mask[0::stride] = False
        mask[1::stride] = False
        batch_negs = embs[mask]

        queue_embs = memory.get()

        loss = info_nce_loss(
            anchors=anchors,
            positives=positives,
            batch_negs=batch_negs,
            queue_embs=queue_embs,
            temperature=cfg.training.temperature,
        )

        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim)
        scaler.update()
        scheduler.step()

        if cfg.wandb.use_wandb:
            wandb.log({"train_loss": loss.item()}, step=global_step + step)
        total_loss += loss.item()
        memory.enqueue(batch_negs)  # в память кладём только негативы
    return total_loss / len(train_dl)


def validate_epoch(model, val_dl, cfg, epoch_idx):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_dl, desc=f"Epoch {epoch_idx}/{cfg.training.epochs} [val]"):
            embs = model(**batch).last_hidden_state[:, 0, :]
            stride = 2 + cfg.batch.num_hard_negs
            anchors = embs[0::stride]
            positives = embs[1::stride]

            mask = torch.ones(embs.size(0), dtype=torch.bool, device=embs.device)
            mask[0::stride] = False
            mask[1::stride] = False
            batch_negs = embs[mask]

            loss = info_nce_loss(
                anchors,
                positives,
                batch_negs,
                queue_embs=torch.empty(0, cfg.model.hidden_dim, device=cfg.device),
                temperature=cfg.training.temperature,
            )
            val_loss += loss.item()
    if cfg.wandb.use_wandb:
        wandb.log({"val_loss": val_loss / len(val_dl)}, step=epoch_idx)
    return val_loss / len(val_dl)

