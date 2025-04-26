import os
import random

import hydra
import torch
from omegaconf import DictConfig

from src.utils.data_utils import get_dataloaders
from src.utils.model_utils import build_tokenizer_and_model, get_optim_and_scheduler
from src.utils.train_utils import train_epoch, validate_epoch
from src.utils.wandb_utils import init_wandb
from src.implementations.cross_batch_memory import CrossBatchMemory

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    if cfg.wandb.use_wandb:
        init_wandb(cfg)

    tokenizer, model = build_tokenizer_and_model(cfg)

    train_dl, val_dl = get_dataloaders(cfg, tokenizer)

    total_steps = cfg.training.epochs * len(train_dl)

    optim, scheduler = get_optim_and_scheduler(cfg, model, total_steps)

    num_batch_negs = cfg.batch.batch_size * cfg.batch.num_hard_negs
    queue_size = max(0, cfg.batch.ref_size - 1 - num_batch_negs)
    print(f"INFO: CrossBatchMemory queue size per process = {queue_size}")

    memory = CrossBatchMemory(queue_size, cfg.model.hidden_dim, cfg.device)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device == "cuda"))

    for epoch in range(1, cfg.training.epochs + 1):
        train_loss = train_epoch(
            model, train_dl, optim, scheduler, memory, scaler, cfg, epoch
        )
        val_loss = validate_epoch(model, val_dl, cfg, epoch)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    out_dir = os.getcwd()
    model_path = os.path.join(out_dir, "model.pt")
    tokenizer_path = os.path.join(out_dir, "tokenizer")
    torch.save(model.state_dict(), model_path)
    tokenizer.save_pretrained(tokenizer_path)
    print(
        f"✅ Saved model to {model_path} and tokenizer to {tokenizer_path}"
    )


if __name__ == "__main__":
    main()