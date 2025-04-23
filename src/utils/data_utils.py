import os
from datasets import load_dataset
from torch.utils.data import DataLoader

from src.implementations.triplet_collator import TripletCollator
from src.implementations.stratified_batch_sampler import StratifiedBatchSampler

def get_dataloaders(cfg, tokenizer):
    # 1) загрузка и сплит
    full = load_dataset(cfg.dataset.dataset_name, split="train")
    full = full.class_encode_column("dataset_name")
    splits = full.train_test_split(
        test_size=cfg.dataset.test_size,
        stratify_by_column="dataset_name",
        seed=cfg.seed,
    )
    train_ds, val_ds = splits["train"], splits["test"]

    # 2) collator + sampler
    collator = TripletCollator(tokenizer, cfg.model.max_len, cfg.device)
    train_ids = train_ds["dataset_name"]
    sampler = StratifiedBatchSampler(train_ids, cfg.batch.batch_size, drop_last=False)

    train_dl = DataLoader(
        train_ds,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=4,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )
    return train_dl, val_dl
