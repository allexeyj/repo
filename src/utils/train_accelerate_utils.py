import torch
import os
import shutil
from tqdm.auto import tqdm
from src.utils.loss_utils import info_nce_loss


def train_epoch_accelerate(accelerator, model, tokenizer, train_dl, optim, scheduler, memory, cfg, epoch_idx):
    model.train()
    total_loss = 0.0
    base_step = (epoch_idx - 1) * len(train_dl)
    for step, batch in enumerate(tqdm(train_dl, desc=f"Epoch {epoch_idx}/{cfg.training.epochs} [train]")):
        with accelerator.accumulate(model):
            with accelerator.autocast():
                embs = model(**batch).last_hidden_state[:, 0, :]

            stride = 2 + cfg.batch.num_hard_negs
            anchors = embs[0::stride]
            positives = embs[1::stride]
            mask = torch.ones(embs.size(0), dtype=torch.bool, device=embs.device)
            mask[0::stride] = mask[1::stride] = False
            batch_negs = embs[mask]

            all_negs = accelerator.gather(batch_negs).detach()  # чтобы CrossBatchMemory делился  между GPU
            memory.enqueue(all_negs)

            loss = info_nce_loss(
                anchors,
                positives,
                batch_negs,
                memory.get(),
                cfg.training.temperature,
            )
            accelerator.backward(loss)
            # градиент-клиппинг выполняет Deepspeed внутри себя
            optim.step()
            optim.zero_grad()
            scheduler.step()

        total_loss += loss.item()

        current_step = base_step + step

        if cfg.wandb.use_wandb and accelerator.is_main_process:
            accelerator.log({"train_loss": loss.item()}, step=current_step)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        epoch_dir = os.path.join(cfg.training.output_dir, f"epoch_{epoch_idx}")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            epoch_dir,
            save_function=accelerator.save
        )
        tokenizer.save_pretrained(epoch_dir)
        accelerator.print(f"Model saved to {epoch_dir}")
    return total_loss / len(train_dl)


def validate_epoch_accelerate(accelerator, model, val_dl, cfg, epoch_idx):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_dl, desc=f"Epoch {epoch_idx}/{cfg.training.epochs} [val]"):
            with accelerator.autocast():
                embs = model(**batch).last_hidden_state[:, 0, :]

            stride = 2 + cfg.batch.num_hard_negs
            anchors = embs[0::stride]
            positives = embs[1::stride]
            mask = torch.ones(embs.size(0), dtype=torch.bool, device=embs.device)
            mask[0::stride] = mask[1::stride] = False
            batch_negs = embs[mask]

            loss = info_nce_loss(
                anchors,
                positives,
                batch_negs,
                torch.empty(0, cfg.model.hidden_dim, device=accelerator.device),
                cfg.training.temperature,
            )
            loss = accelerator.gather(loss).mean()
            val_loss += loss.item()

    avg_loss = val_loss / len(val_dl)
    if cfg.wandb.use_wandb and accelerator.is_main_process:
        accelerator.log({"val_loss": avg_loss}, step=epoch_idx)
    return avg_loss
