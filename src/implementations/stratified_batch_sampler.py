import random
import numpy as np
from torch.utils.data import Sampler
from typing import List

class StratifiedBatchSampler(Sampler[List[int]]):
    """
    Формирует батчи так, чтобы внутри каждого batch_size
    все примеры были из одного источника (dataset_id).

    Сделано на основе статьи:
    В статье сказано, что они формируют каждый device-batch (мини-батч) из примеров только одного датасета,
    а затем в глобальный батч объединяют мини-батчи с разных GPU.
    Такой подход улучшает разнообразие источников в батче и при этом избегает утечки hard-негативов между устройствами
    """

    def __init__(self, dataset_ids: List[int], batch_size: int, drop_last: bool = False):
        """
        Args:
            dataset_ids (List[int]): список размером N, где dataset_ids[i] — id датасета для i-го примера
            batch_size (int): размер каждого мини-батча
            drop_last (bool): опустить последний неполный батч в каждой группе
        """
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Группируем индексы по id датасета
        self.groups = {}
        for idx, ds in enumerate(dataset_ids):
            self.groups.setdefault(ds, []).append(idx)
        # Для семплинга групп берём вероятности пропорционально их размерам
        sizes = np.array([len(v) for v in self.groups.values()], dtype=float)
        self.group_ids = list(self.groups.keys())
        self.probs = sizes / sizes.sum()

        # Копии списков индексов для вытаскивания и перемешивания
        self.buffers = {ds: indices.copy() for ds, indices in self.groups.items()}
        for buf in self.buffers.values():
            random.shuffle(buf)

    def __iter__(self):
        # Бесконечный генератор батчей; остановку сделает DataLoader
        while True:
            # Выбираем группу по вероятности
            ds = random.choices(self.group_ids, weights=self.probs, k=1)[0]
            buf = self.buffers[ds]

            # Если в буфере недостаточно, либо перезаполняем и перетасовываем,
            # либо, если drop_last=True, пропускаем этот батч
            if len(buf) < self.batch_size:
                if self.drop_last:
                    continue
                # заново заполнить и перемешать
                buf.extend(self.groups[ds])
                random.shuffle(buf)

            # Забираем батч
            batch = [buf.pop() for _ in range(self.batch_size)]
            yield batch

    def __len__(self):
        # Для PyTorch надо вернуть хотя бы число батчей за эпоху
        # Можно определить как total_samples // batch_size
        total = sum(len(v) for v in self.groups.values())
        return total // self.batch_size