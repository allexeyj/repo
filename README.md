# 🛠 ru-en-embedder-finetune 🚀

> 🎉 Supervised Fine-Tuning эмбеддера для русско-английских запросов после weekly SFT стадии.

---

## 🗂 Датасеты
Данные доступны на Hugging Face Hub: https://huggingface.co/datasets/Alexator26/triplets-3M-class-encode

| Датасет                                              | Язык | Размер    |
|------------------------------------------------------|:----:|----------:|
| nomic-ai/nomic-embed-supervised-data                 | en   | 1,668,794 |
| sentence-transformers/all-nli                        | ru   |     8,434 |
| nyuuzyou/fishkinet-posts                             | ru   |    99,996 |
| IlyaGusev/gazeta                                     | ru   |    25,120 |
| its5Q/habr_qna                                       | ru   |   100,000 |
| zloelias/lenta-ru                                    | ru   |   100,000 |
| Shitao/bge-m3-data (miracl_ru_filtered)              | ru   |     4,680 |
| Shitao/bge-m3-data (mrtydi_ru_filtered_triplts)      | ru   |     5,349 |
| query2doc_msmarco                                    | ru   |   380,000 |
| deepvk/ru-HNP                                        | ru   |   100,000 |
| deepvk/ru-WANLI                                      | ru   |    32,253 |
| wikimedia/wikipedia                                  | ru   |    80,000 |
| CarlBrendt/Summ_Dialog_News                          | ru   |    40,000 |
| RussianNLP/wikiomnia                                 | ru   |    17,064 |
| ts5Q/yandex-q                                        | ru   |   100,000 |
| IlyaGusev/ficbook                                    | ru   |    80,000 |
| IlyaGusev/ru_stackoverflow                           | ru   |    50,000 |
| kuznetsoffandrey/sberquad                            | ru   |    27,016 |
| IlyaGusev/saiga_scored                               | ru   |    17,695 |

**Итого:**

|       |     en   |      ru   |     всего   |
|-------|---------:|----------:|------------:|
| **Sum** | 1,668,794 | 1,267,607 | 2,936,401  |

---

## 🌟 Структура проекта

```bash
├── README.md
├── configs
│   ├── accelerate
│   │   └── default_config.yaml
│   ├── config.yaml
│   ├── dataset
│   │   └── triplets_3M.yaml
│   └── model
│       └── small.yaml
├── notebooks
│   └── semi_hard_negs_mining.ipynb
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── implementations
│   │   ├── cross_batch_memory.py
│   │   ├── stratified_batch_sampler.py
│   │   └── triplet_collator.py
│   └── utils
│       ├── data_utils.py
│       ├── loss_utils.py
│       ├── model_utils.py
│       ├── train_accelerate_utils.py
│       ├── train_utils.py
│       └── wandb_utils.py
├── train.py
└── train_accelerate.py
```

---

## 🚀 Быстрый старт

1. **Установите зависимости**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Выполните ускоренный запуск** (с Deepspeed, mixed-precision и др.):
   ```bash
   export WANDB_API_KEY="<ваш ключ>"
   accelerate launch \
     --config_file configs/accelerate/default_config.yaml \
     train_accelerate.py
   ```

---

## 🧩 Как это работает

1. **DataLoader + StratifiedBatchSampler** — батчи из одного источника, чтобы активно учить hard negatives.
2. **TripletCollator** — формирует `(q, pos, negs…)` с токенами-префиксами `search_query`, `search_document`.
3. **CrossBatchMemory** — очередь эмбеддингов предыдущих негативов для увеличения эффективного батча.
4. **InfoNCE Loss** — стандартный, но с дополнительными негативами из памяти.

---


