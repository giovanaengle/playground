from abc import ABC, abstractmethod
from dataclasses import field
from pathlib import Path
from typing import Any

from common import Config
from data import Annotations, TaskType


class Model(ABC):
    classes: dict[int, str] = field(default_factory=lambda: {})
    model: Any | None = None

    def __init__(self, config: Config) -> None:
        self.input = Path(config.path('input'))
        if not self.input.exists():
            raise Exception(f'Model path does not exist: {self.input}')

        self.config = config
        self.architecture = config.str('architecture')
        self.data = self.config.path('data')
        self.output = Path(config.path('output'))
        self.params = Config(path=config.path('params'))
        self.task = TaskType.from_str(config.str('task'))
        self.weights = config.str('weights')

        self.output.mkdir(exist_ok=True, parents=True)
        self._set()

    def _set(self) -> None:
        self.load()
        self.info() 

    @abstractmethod
    def categories(self) -> None:
        raise NotImplementedError
   
    @abstractmethod
    def evaluate(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def export(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def info(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: Any):
        raise NotImplementedError

    @abstractmethod
    def to_annotations(self, results: Any) -> Annotations:
        raise NotImplementedError
    
    @abstractmethod
    def train(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def validate(self) -> Any:
        raise NotImplementedError
