**ru-en-embedder-finetune** 🚀

&#x20; &#x20;

> 🎉 Делаем SFT(supervised fine-tuning) эмбеддера для русско-английских запросов после weekly SFT стадии.

---

## 🗂 Датасеты
Данные доступны на Hugging Face Hub: https://huggingface.co/datasets/Alexator26/triplets-3M-class-encode

| Датасет                                              | Язык | Размер    |
|-------------------------------------------------------|:----:|----------:|
| nomic-ai/nomic-embed-supervised-data                  | en   | 1,668,794 |
| sentence-transformers/all-nli                         | ru   |     8,434 |
| nyuuzyou/fishkinet-posts                              | ru   |    99,996 |
| IlyaGusev/gazeta                                      | ru   |    25,120 |
| its5Q/habr_qna                                        | ru   |   100,000 |
| zloelias/lenta-ru                                     | ru   |   100,000 |
| Shitao/bge-m3-data (miracl_ru_filtered)               | ru   |     4,680 |
| Shitao/bge-m3-data (mrtydi_ru_filtered_triplts)       | ru   |     5,349 |
| query2doc_msmarco                                     | ru   |   380,000 |
| deepvk/ru-HNP                                         | ru   |   100,000 |
| deepvk/ru-WANLI                                       | ru   |    32,253 |
| wikimedia/wikipedia                                   | ru   |    80,000 |
| CarlBrendt/Summ_Dialog_News                           | ru   |    40,000 |
| RussianNLP/wikiomnia                                  | ru   |    17,064 |
| ts5Q/yandex-q                                         | ru   |   100,000 |
| IlyaGusev/ficbook                                     | ru   |    80,000 |
| IlyaGusev/ru_stackoverflow                            | ru   |    50,000 |
| kuznetsoffandrey/sberquad                             | ru   |    27,016 |
| IlyaGusev/saiga_scored                                | ru   |    17,695 |

|       |     en   |      ru   |     всего   |
|-------|---------:|----------:|------------:|
| Итого | 1,668,794 | 1,267,607 | 2,936,401  |



## 🌟 Структура проекта

```bash
├── README.md
├── environment.yml        # конда-окружение
├── requirements.txt       # pip-зависимости
├── accelerate/            # конфиг Accelerate/Deepspeed
├── configs/               # Hydra-конфиги (model, dataset)
├── src/                   # Исходники
│   ├── implementations/   # Collators, Samplers, Memory
│   └── utils/             # Загрузка данных, тренировка, WandB и т.д.
├── train.py               # Обычный трейнинг (amp)
└── train_accelerate.py    # Запуск через Accelerate
```

---

## 🚀 Быстрый старт

1. **configs/model/base.yaml**:
   - `_model_`: имя модели (например, `deepvk/USER2-base`)
   - `revision`: ревизия (например, `weakly_sft`)
   - `max_len`: максимальная длина токенов
2. **configs/dataset/triplets.yaml**:
   - `dataset_path`: путь к локальной папке с датасетом или
   - `dataset_name`: имя датасета на HF (оставить `null` одну из опций)
   - `test_size`: доля валидационного сплита (например, `0.02`)
3. **accelerate/default_config.yaml**:
   - `compute_environment`, `distributed_type`, `mixed_precision`, `num_processes`, `gpu_ids`, `deepspeed_config` и др.
4. **configs/config.yaml**:
   - подключает нужные модели и датасеты через `defaults`
   - задаёт `seed`, `device`, параметры `batch`, `training`, `wandb`, `resume_from`


Запуск обучения с accelerate:
```bash
accelerate launch train_accelerate.py
```


## 🧩 Как это работает

1. **DataLoader** с `StratifiedBatchSampler`  — батчи из одного источника, чтобы не утекали hard negative между процессами.
2. **TripletCollator** — конвертирует тройки `(q, pos, negs…)` в токены с префиксами `search_query: …`, `search_document: …`.
3. **CrossBatchMemory** — очередь эмбеддингов предыдущих негативов для более стабильного InfoNCE(нужен большой батч, а большой батч наши видеокарты не тянут, очередь "симулирует" большой батч).
4. **InfoNCE loss** — стандарт, единственное, берутся дополнительные негативы из очереди.
5. Сохраняем чекпоинты каждые `ckpt_steps` шагов, логируем в W&B.
---
