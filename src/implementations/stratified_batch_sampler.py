import random
import numpy as np
from torch.utils.data import Sampler
from typing import List

class StratifiedBatchSampler(Sampler[List[int]]):
    """
    Формирует батчи так, чтобы внутри каждого batch_size
    все примеры были из одного источника (dataset_id).

    device-батчи формируются из одного датасета,
    а затем объединяются в глобальный батч. Это улучшает
    разнообразие источников и предотвращает утечку hard-негативов.
    """

    def __init__(self, dataset_ids: List[int], batch_size: int): #drop_last по сути всегда True
        """
        Args:
            dataset_ids (List[int]): список длины N, где dataset_ids[i] — id датасета для i-го примера
            batch_size (int): размер каждого device-батча
        """
        self.batch_size = batch_size

        # Группируем индексы по id датасета
        self.groups = {}
        for idx, ds in enumerate(dataset_ids):
            self.groups.setdefault(ds, []).append(idx)

        # Вероятности выбора группы пропорциональны её размеру
        sizes = np.array([len(v) for v in self.groups.values()], dtype=float)
        self.group_ids = list(self.groups.keys())
        self.probs = sizes / sizes.sum()

    def __iter__(self):
        # 1) Общее число батчей в эпохе
        n_batches = len(self)

        # 2) Локальные буферы для каждой группы, перемешанные заново
        buffers = {ds: idxs.copy() for ds, idxs in self.groups.items()}
        for buf in buffers.values():
            random.shuffle(buf)

        # 3) Генерируем ровно n_batches батчей
        for _ in range(n_batches):
            # Выбираем группу по распределению
            ds = random.choices(self.group_ids, weights=self.probs, k=1)[0]
            buf = buffers[ds]

            # Если осталось меньше batch_size, рефилл и перемешивание
            if len(buf) < self.batch_size:
                buf.extend(self.groups[ds])
                random.shuffle(buf)

            # Формируем батч и отдаём
            batch = [buf.pop() for _ in range(self.batch_size)]
            yield batch

    def __len__(self) -> int:
        # Определяем число батчей за эпоху как целочисленное деление
        total = sum(len(v) for v in self.groups.values())
        return total // self.batch_size
