import os
import torch

from accelerate import Accelerator
from accelerate.utils import set_seed
import hydra
from omegaconf import DictConfig

from src.utils.data_utils import get_dataloaders
from src.utils.model_utils import build_tokenizer_and_model, get_optim_and_scheduler
from src.implementations.cross_batch_memory import CrossBatchMemory
from src.utils.train_accelerate_utils import train_epoch_accelerate, validate_epoch_accelerate
from src.utils.wandb_utils import init_wandb


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # ─── 0) До всякой инициализации torch — фиксируем hash-seed и cublas workspace
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # ─── 1) Создаём Accelerator (читать mixed_precision из accelerate/default_config.yaml)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=None,  # None → возьмётся из вашего accelerate/default_config.yaml
        log_with="wandb"  # логируем в W&B
    )

    # ─── 2) Сиды и детерминизм
    set_seed(cfg.seed, device_specific=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # ─── 3) WandB
    if cfg.wandb.use_wandb and accelerator.is_main_process:
        init_wandb(cfg)

    # ─── 4) Строим токенизатор и модель
    tokenizer, model = build_tokenizer_and_model(cfg)

    # ─── 5) Даталоадеры
    train_dl, val_dl = get_dataloaders(cfg, tokenizer)

    # ─── 6) Оптимизатор и шедулер
    total_steps = cfg.training.epochs * len(train_dl)
    optim, scheduler = get_optim_and_scheduler(cfg, model, total_steps)

    # ─── 7) Cross-Batch-Memory (глобальный размер очереди = ref_size – batch*(1+negs) на все процессы)
    global_batch = cfg.batch.batch_size * (1 + cfg.batch.num_hard_negs) * accelerator.num_processes
    queue_size = max(0, cfg.batch.ref_size - global_batch) * accelerator.num_processes
    memory = CrossBatchMemory(queue_size, cfg.model.hidden_dim, accelerator.device)

    # ─── 8) Регистрируем stateful-объекты для корректного сохранения/загрузки
    accelerator.register_state(sampler=train_dl.batch_sampler, memory=memory)

    # ─── 9) Готовим всё к распараллеливанию
    model, optim, train_dl, val_dl, scheduler = accelerator.prepare(
        model, optim, train_dl, val_dl, scheduler
    )

    if cfg.resume_from:
        accelerator.load_state(cfg.resume_from)

    # 10) Цикл обучения и валидации через вынесенные функции

    for epoch in range(1, cfg.training.epochs + 1):
        train_loss = train_epoch_accelerate(
            accelerator, model, train_dl, optim, scheduler, memory, cfg, epoch
        )
        val_loss = validate_epoch_accelerate(
            accelerator, model, val_dl, cfg, epoch
        )

    if accelerator.is_main_process:
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")


if __name__ == "__main__":
    main()
