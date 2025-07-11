from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
from shutil import rmtree

from common import Config
from data import Data
from dataset.hf import HFDataset
from dataset.ul import ULDataset


class DatasetFormat(Enum):
    CLASSIFY = 'classify'
    DETECT = 'detect'
    POSE = 'pose'
    SEGMENT = 'segment'

    @staticmethod
    def from_str(s: str) -> 'DatasetFormat':
        dfs: list[DatasetFormat] = [
            DatasetFormat.CLASSIFY,
            DatasetFormat.DETECT,
            DatasetFormat.POSE,
            DatasetFormat.SEGMENT,
        ]
        for df in dfs:
            if s == df.value:
                return df

        raise Exception(f'unknown dataset format: {s}')

class Dataset:
    test: list[Data] = []
    train: list[Data] = []
    valid: list[Data] = []

    def __init__(self, config: Config) -> None:
        classes = config.strs('classes'),
        classes = [cls.lower() for cls in classes]

        self.balance = config.bool('balance')
        self.classes = classes
        self.format = DatasetFormat.from_str(config.str('task'))
        self.path = config.path('path')
        self.split = config.floats('split') 

    @abstractmethod
    def get(self) -> None:
        raise NotImplementedError

    def prepare(self, data: list[Data]) -> None:
        counters: dict[int, int] = {}
        for d in data:
            for anno in d.annotations.items:
                if not anno.class_id in counters:
                    counters[anno.class_id] = 1
                else:
                    counters[anno.class_id] += 1

        print(f'Dataset distribution: {counters}')

        # Balance
        if self.balance:
            min: int = math.inf
            for count in counters.values():
                if count is None:
                    continue
                if count < min:
                    min = count

            for class_id in counters.keys():
                counters[class_id] = min

            print(f'Dataset distribution after balancing: {counters}')

        # Split
        if len(self.split) == 3:
            items: dict[int, list[Data]] = {}
            for class_id in counters.keys():
                items[class_id] = []

            for d in data:
                for anno in d.annotations.items:
                    items[anno.class_id].append(d.copy())

            for class_id, count in counters.items():
                test = math.floor(count * self.split[0])
                for i in range(test):
                    if i == test:
                        break
                    self.test.append(items[class_id].pop())

                train = math.ceil(count * self.split[1])
                for i in range(train):
                    if i == train:
                        break
                    self.train.append(items[class_id].pop())

                valid = count - (test + train)
                for i in range(valid):
                    if i == valid:
                        break
                    self.valid.append(items[class_id].pop())

    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError

    def setup(self) -> None:
        rmtree(self.path, ignore_errors=True, onexc=None)
        self.path.mkdir(exist_ok=True, parents=True)

class DatasetFactory:
    datasets: dict[str, Dataset] = {
        'hugging_face': HFDataset,
        'ultralytics': ULDataset,
    }

    @staticmethod
    def create(config: Config) -> Dataset:
        name = config.str('framework')
        if name in DatasetFactory.datasets:
            return DatasetFactory.datasets[name](config)
        else:
            raise Exception(f'Dataset framework not implemented: {name}')