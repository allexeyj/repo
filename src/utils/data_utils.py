from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
import random
import numpy as np
import torch

from src.implementations.triplet_collator import TripletCollator
from src.implementations.stratified_batch_sampler import StratifiedBatchSampler


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(cfg, tokenizer):
    # Загрузка датасета: либо по имени на HF Hub, либо из локальной папки
    name = cfg.dataset.get("dataset_name")
    path = cfg.dataset.get("dataset_path")
    if (name and path) or (not name and not path):
        raise ValueError("Необходимо указать ровно одно из 'dataset_name' или 'dataset_path' в конфиге.")
    if name:
        full = load_dataset(name, split="train")
    else:
        full = load_from_disk(path, keep_in_memory=True)['train'] #так как может быть read only

    full = full.class_encode_column("dataset_name")
    splits = full.train_test_split(
        test_size=cfg.dataset.test_size,
        stratify_by_column="dataset_name",
        seed=cfg.seed,
    )
    train_ds, val_ds = splits["train"], splits["test"]

    collator = TripletCollator(tokenizer, cfg.model.max_len)
    train_ids = train_ds["dataset_name"]
    sampler = StratifiedBatchSampler(train_ids, cfg.batch.batch_size, drop_last=False)


    g = torch.Generator().manual_seed(cfg.seed)

    train_dl = DataLoader(
        train_ds,
        batch_sampler=sampler,
        collate_fn=collator,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch.batch_size,
        shuffle=False,
        collate_fn=collator,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_dl, val_dl
