from abc import ABC, abstractmethod
from dataclasses import field
from pathlib import Path
from typing import Any

from common import Config
from data import Annotations


class Model(ABC):
    classes: dict[int, str] = field(default_factory=lambda: {})
    model: Any | None = None

    def __init__(self, config: Config) -> None:
        self.config = config

        self.name = config.str('architecture')
        self.weights = config.str('weights')
        self.path = Path(config.str('path'))
        if not self.path.exists():
            raise Exception(f'Model path does not exist: {self.path}')

        self._set()

    def _set(self) -> None:
        self.load()
        self.categories()
        self.info() 

    @abstractmethod
    def _to_annotation(self) -> Annotations:
        NotImplemented 

    @abstractmethod
    def categories(self) -> None:
        NotImplemented

    @abstractmethod
    def info(self) -> None:
        NotImplemented

    @abstractmethod
    def load(self) -> None:
        NotImplemented

    @abstractmethod
    def predict(self) -> Annotations:
        NotImplemented

    @abstractmethod
    def to_dataset(self) -> None:
        NotImplemented

    @abstractmethod
    def train(self) -> Any:
        NotImplemented

    @abstractmethod
    def validate(self) -> Any:
        NotImplemented
