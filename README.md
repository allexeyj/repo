
# Text Embedders via Contrastive Learning

## Описание

Проект посвящён созданию эмбеддера для русского языка на основе материалов из коллекции [Sentence Encoders](https://huggingface.co/collections/deepvk/sentence-encoders-6667222a68458ec9acfea9fb), с последующим применением в разных задачах(например, STS, Retrival, Classification).

## Участники команды

- Куценко Дмитрий  
- Самсанович Екатерина  
- Шатурный Алексей  

## Конкуренты и аналоги

Проект носит учебный характер и не ставит цель конкурировать с промышленными решениями. Схожие системы уже существуют — см. [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

## MVP (минимально жизнеспособный продукт)

- Обученная модель эмбеддера.  
- Семантический поиск по текстам.  
- Оценка на стандартных бенчмарках(ruMTEB).  
- Сравнение с другими эмбеддерами.

## Решение

- В качестве базовой модели используется `deepvk/RuModernBERT-base`.  
- Датасет `ru-HNP` был адаптирован под нужный формат: к каждому положительному примеру добавлен случайный отрицательный.  
- Обучение ведётся с использованием contrastive learning и AnglE loss.

### Запуск обучения

```bash
CUDA_VISIBLE_DEVICES=0 angle-trainer \
  --model_name_or_path deepvk/RuModernBERT-base \
  --train_name_or_path Alexator26/ru-hnp-renamed-and-fixed-500k-v3 \
  --train_split_name train \
  --valid_name_or_path Alexator26/ru-hnp-renamed-and-fixed-500k-v3 \
  --valid_split_name validation \
  --valid_name_or_path_for_callback Alexator26/eval-ru-sts-dataset \
  --valid_split_name_for_callback train \
  --save_dir ./angle-rumodernbert-ru-hnp-output \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 3 \
  \
  --eval_strategy steps \
  --eval_steps 200 \
  \
  --logging_steps 50 \
  --epochs 3 \
  --batch_size 4 \
  --gradient_accumulation_steps 16 \
  --fp16 1 \
  --seed 42 \
  \
  --pooling_strategy cls \
  --maxlen 512 \
  \
  --wandb_project angle-ru-hnp-experiments \
  --wandb_log_model false
```

## Метрики на ruMTEB
zero-shot - 100%, т.е без дообучения на трейн сплитах бенчмарков

![ruMTEB](./ru-mteb.png)

Метрики по задачам `MIRACLReranking`, `MIRACLRetrieval`, `MassiveIntentClassification`, `MassiveScenarioClassification`, `TERRa` не учитывались — их не оценивали.


## Ограничения по ресурсам

Проект максимально использует доступные ресурсы — обучение проводится на бесплатной GPU-сессии Kaggle, которая ограничена 12 часами. Уже сейчас обучение едва помещается в этот лимит, и дальнейшие эксперименты проводить сложно.

Например, мы не можем добавить даже относительно компактный(около 580к триплетов) датасет [all-nli](https://huggingface.co/datasets/sentence-transformers/all-nli) в трейн. Нет ресурсов на перевод датасета на русский и последующее обучение. **Без дополнительных ресурсов вряд-ли получится сильно учучшить результат**
