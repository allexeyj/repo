import os
import torch
from hydra.utils import get_original_cwd
import random

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate import DataLoaderConfiguration

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.data_utils import get_dataloaders
from src.utils.model_utils import build_tokenizer_and_model, get_optim_and_scheduler
from src.implementations.cross_batch_memory import CrossBatchMemory
from src.utils.train_accelerate_utils import train_epoch_accelerate, validate_epoch_accelerate

def set_seed(seed: int) -> None:
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    accelerator = Accelerator(
        log_with="wandb",
        project_dir=cfg.training.output_dir)

    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = cfg.batch.batch_size
    accelerator.state.deepspeed_plugin.deepspeed_config['gradient_accumulation_steps'] = 1

    # ─── 1.1) Инициализируем трекеры (имя run, параметры эксперимента)
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=cfg.wandb.project,  # уникальное имя запуска
            config=OmegaConf.to_container(cfg, resolve=False)
        )

    # ─── 3) Строим токенизатор и модель
    tokenizer, model = build_tokenizer_and_model(cfg)

    # ─── 4) Даталоадеры
    train_dl, val_dl = get_dataloaders(cfg, tokenizer)

    # ─── 5) Оптимизатор и шедулер
    total_steps = cfg.training.epochs * len(train_dl)
    optim, scheduler = get_optim_and_scheduler(cfg, model, total_steps)

    # ─── 6) Cross-Batch-Memory
    # ref_size = 1 (позитив) + N_batch_negs + N_queue_negs; ref_size - имитируемый батч
    num_batch_negs = cfg.batch.batch_size * cfg.batch.num_hard_negs
    queue_size = max(0, cfg.batch.ref_size - 1 - num_batch_negs)
    accelerator.print(f"INFO: CrossBatchMemory queue size per process = {queue_size}")
    memory = CrossBatchMemory(int(queue_size), cfg.model.hidden_dim, accelerator.device)

    # ─── 8) Готовим всё к распараллеливанию
    model, optim, train_dl, val_dl, scheduler = accelerator.prepare(
        model, optim, train_dl, val_dl, scheduler
    )

    # 9) Цикл обучения и валидации через вынесенные функции

    for epoch in range(1, cfg.training.epochs + 1):
        train_loss = train_epoch_accelerate(
            accelerator, model, train_dl, optim, scheduler, memory, cfg, epoch
        )
        val_loss = validate_epoch_accelerate(
            accelerator, model, val_dl, cfg, epoch
        )

        if accelerator.is_main_process:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    accelerator.wait_for_everyone()  # Дождаться всех процессов перед завершением
    if accelerator.is_main_process:
        accelerator.end_training()


if __name__ == "__main__":
    set_seed(cfg.seed)
    main()
