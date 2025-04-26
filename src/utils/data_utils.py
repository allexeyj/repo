from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
import random
import numpy as np
import torch

from src.implementations.triplet_collator import TripletCollator
from src.implementations.stratified_batch_sampler import StratifiedBatchSampler



def get_dataloaders(cfg, tokenizer):
    # Загрузка датасета: либо по имени на HF Hub, либо из локальной папки
    name = cfg.dataset.get("dataset_name")
    path = cfg.dataset.get("dataset_path")
    if (name and path) or (not name and not path):
        raise ValueError("Необходимо указать ровно одно из 'dataset_name' или 'dataset_path' в конфиге.")
    if name:
        full = load_dataset(name, split="train")
    else:
        full = load_from_disk(path)['train']

    full = full.select(range(1000))

    splits = full.train_test_split(
        test_size=cfg.dataset.test_size,
        stratify_by_column="dataset_name",
        seed=cfg.seed,
    )
    train_ds, val_ds = splits["train"], splits["test"]

    collator = TripletCollator(tokenizer, cfg.model.max_len)
    train_ids = train_ds["dataset_name"]



    if cfg.batch.use_stratified_batch_sampler:
        sampler = StratifiedBatchSampler(train_ids, cfg.batch.batch_size)  # drop_last по сути всегда True
        train_dl = DataLoader(
            train_ds,
            batch_sampler=sampler,
            collate_fn=collator,
            pin_memory=True,
            prefetch_factor=cfg.batch.prefetch_factor,
            num_workers=cfg.batch.num_worker
        )
        print('custom sampler')
    else:
        train_dl = DataLoader(
            train_ds,
            batch_size=cfg.batch.batch_size,
            shuffle=True,
            collate_fn=collator,
            pin_memory=True,
            prefetch_factor=cfg.batch.prefetch_factor,
            num_workers=cfg.batch.num_worker,
            drop_last=cfg.batch.drop_last
        )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch.batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=True,
        prefetch_factor=cfg.batch.prefetch_factor,
        num_workers=cfg.batch.num_workers,
        drop_last=cfg.batch.drop_last

    )

    return train_dl, val_dl
