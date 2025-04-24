import os
import torch
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW


def build_tokenizer_and_model(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model._model_, revision=cfg.model.revision)
    model = AutoModel.from_pretrained(cfg.model._model_, revision=cfg.model.revision)
    model.gradient_checkpointing_enable()
    # полагайемся на accelerator.prepare для размещения модели и тензоров на нужном GPU.
    return tokenizer, model


def get_optim_and_scheduler(cfg, model, total_steps):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n.lower() for nd in ["bias", "layernorm", "ln_", ".norm"]):
            no_decay.append(p)
        else:
            decay.append(p)

    optim = AdamW(
        [{"params": decay, "weight_decay": cfg.training.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.training.lr,
    )
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
    )
    return optim, scheduler
