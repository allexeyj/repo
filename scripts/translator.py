#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chunk-wise translation with incremental pushes to HF Hub.

Колонки:
  • query      : str
  • positive   : str
  • negative   : List[str]  (ровно 5)

Все прочие колонки сохраняются без изменений.

Параметры:
  --push-every N   → после каждых N переведённых примеров пушим буфер.

Запуск на кагле:

%%bash
# ---------- первый процесс на GPU-0 ----------
python script.py \
  --src-dataset your_user/your_dataset \
  --chunk-size 1000 \
  --start-idx   0 \
  --batch-txt   16 \
  --push-to     your_user/translated_dataset \
  --device      cuda:0 &          # ← амперсанд ставит в фон

# ---------- второй процесс на GPU-1 ----------
python script.py \
  --src-dataset your_user/your_dataset \
  --chunk-size 1000 \
  --start-idx   1 \
  --batch-txt   16 \
  --push-to     your_user/translated_dataset \
  --device      cuda:1 &

wait   # ждём, пока оба фоновых процесса завершатся

"""



import argparse
import itertools
import os
import shutil
import tempfile
from typing import Any, Dict, List

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from huggingface_hub import Repository

# --------------------------------------------------------------------- #
#                         КОНСТАНТЫ                                     #
# --------------------------------------------------------------------- #
HF_TOKEN = os.environ["HF_TOKEN"] 
MODEL_NAME = "facebook/nllb-200-3.3B"
SRC_LANG   = "eng_Latn"
TGT_LANG   = "rus_Cyrl"
FP16       = True


# --------------------------------------------------------------------- #
#                        ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ                        #
# --------------------------------------------------------------------- #
def assert_power_of_two(x: int) -> None:
    assert x > 0 and (x & (x - 1)) == 0, f"BATCH size ({x}) must be a power of two"


def chunk_iter(it: List[Any], size: int):
    for i in range(0, len(it), size):
        yield it[i : i + size]


def translate_list(texts: List[str], translator, batch_txt: int) -> List[str]:
    """Перевести произвольный список строк, режем его на куски batch_txt."""
    out: List[str] = []
    for part in chunk_iter(texts, batch_txt):
        tr = translator(part)
        out.extend(r["translation_text"] for r in tr)
    return out


def translate_batch(batch: Dict[str, Any], translator, batch_txt: int) -> Dict[str, Any]:
    queries_tr   = translate_list(batch["query"],    translator, batch_txt)
    positives_tr = translate_list(batch["positive"], translator, batch_txt)

    flat_neg   = list(itertools.chain.from_iterable(batch["negative"]))
    flat_neg_tr = translate_list(flat_neg, translator, batch_txt)

    n_per_ex = len(batch["negative"][0])
    neg_tr = [
        flat_neg_tr[i * n_per_ex : (i + 1) * n_per_ex] for i in range(len(queries_tr))
    ]

    return {"query": queries_tr, "positive": positives_tr, "negative": neg_tr}


def safe_translate_batch(
    batch: Dict[str, Any], translator, batch_txt: int
) -> Dict[str, Any]:
    try:
        return translate_batch(batch, translator, batch_txt)
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            bs = len(batch["query"])
            if bs <= 1:
                raise
            mid = bs // 2
            first  = {k: v[:mid] for k, v in batch.items()}
            second = {k: v[mid:] for k, v in batch.items()}
            tr1 = safe_translate_batch(first,  translator, batch_txt)
            tr2 = safe_translate_batch(second, translator, batch_txt)
            return {
                "query":    tr1["query"]    + tr2["query"],
                "positive": tr1["positive"] + tr2["positive"],
                "negative": tr1["negative"] + tr2["negative"],
            }
        raise


# --------------------------------------------------------------------- #
#                                MAIN                                   #
# --------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser("Dataset translator with incremental push.")
    parser.add_argument("--src-dataset", type=str, required=True)
    parser.add_argument("--chunk-size",  type=int, required=True)
    parser.add_argument("--start-idx",   type=int, required=True)
    parser.add_argument("--batch-txt",   type=int, required=True)
    parser.add_argument("--push-every",  type=int, default=10_000,
                        help="Сколько переведённых примеров держать в буфере "
                             "перед пушем (по умолчанию 10 000).", required=True)
    parser.add_argument("--push-to",     type=str, required=True,
                        help="huggingface repo, e.g. username/my_dataset")
    parser.add_argument("--device",      type=str, required=True,
                        help="cuda:0 / cuda:1 ...")
    args = parser.parse_args()

    # sanity
    assert_power_of_two(args.batch_txt)
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA not available!")
    device_idx = int(args.device.split(":")[-1])
    if device_idx >= torch.cuda.device_count():
        raise ValueError(f"Requested {args.device}, "
                         f"but only {torch.cuda.device_count()} visible")

    # ----------------------------------------------------------------- #
    #                 загрузка модели и пайплайна                       #
    # ----------------------------------------------------------------- #
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if FP16 else torch.float32,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
    ).eval()

    translator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
        batch_size=args.batch_txt,
        max_length=1024,
    )

    # ----------------------------------------------------------------- #
    #                        DATASET ЧАНК                               #
    # ----------------------------------------------------------------- #
    ds = load_dataset(args.src_dataset, split="train")
    total = len(ds)
    start = args.start_idx * args.chunk_size
    end   = min(start + args.chunk_size, total)
    if start >= total:
        raise ValueError(f"Start {start} ≥ dataset length {total}")
    print(f"[INFO] rows {start}:{end} (count={end-start}) on {args.device}")

    subset = ds.select(range(start, end))

    # ----------------------------------------------------------------- #
    #                подготовка локального git-клона                    #
    # ----------------------------------------------------------------- #
    repo_dir = tempfile.mkdtemp(prefix="hf_repo_")
    repo = Repository(local_dir=repo_dir,
                      clone_from=args.push_to,
                      token=HF_TOKEN,
                      repo_type="dataset")

    # ----------------------------------------------------------------- #
    #                     ПЕРЕВОД + ИНКРЕМЕНТАЛЬНЫЙ PUSH                #
    # ----------------------------------------------------------------- #
    buffer: List[Dict[str, Any]] = []
    pushed_chunks = 0

    def flush_buffer():
        nonlocal pushed_chunks, buffer
        if not buffer:
            return
        chunk_ds = Dataset.from_list(buffer)
        shard_name = f"train_chunk_{args.start_idx}_{pushed_chunks}.parquet"
        shard_path = os.path.join(repo_dir, shard_name)
        chunk_ds.to_parquet(shard_path)

        repo.git_add(shard_name)
        repo.git_commit(f"add {shard_name} ({len(buffer)} rows)")
        repo.git_push()
        print(f"[INFO] pushed {len(buffer)} rows -> {shard_name}")

        buffer = []
        pushed_chunks += 1

    # обработка батчами такого же размера, как и у переводчика
    for i in range(0, len(subset), args.batch_txt):
        raw_batch = subset[i : i + args.batch_txt]

        # --- перевод ---
        # raw_batch – объект Dataset; приведём к dict-of-lists
        raw_dict = raw_batch
        tr = safe_translate_batch(raw_dict, translator, args.batch_txt)

        # --- склеиваем перевод + дополнительные колонки ---
        batch_size = len(tr["query"])
        for j in range(batch_size):
            rec = {k: raw_dict[k][j] for k in raw_dict.keys()
                                      if k not in ("query", "positive", "negative")}
            rec.update(
                query    = tr["query"][j],
                positive = tr["positive"][j],
                negative = tr["negative"][j],
            )
            buffer.append(rec)

        # --- пуш, если набрали N ---
        if len(buffer) >= args.push_every:
            flush_buffer()

    # допушиваем остаток
    flush_buffer()
    print("[INFO] done")

    # подчистим tmp-директорию (по желанию)
    shutil.rmtree(repo_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
