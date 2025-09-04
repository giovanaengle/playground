from dataclasses import dataclass
from typing import Any

from play.common import Config
from play.data import Annotations

from .model import Model


@dataclass
class HfModel(Model):
    def __init__(self, config: Config):
        super().__init__(config)

    def _set(self) -> None:
        super()._set()

    def _to_annotation(self) -> Annotations:
        raise NotImplementedError 

    def categories(self) -> None:
        raise NotImplementedError

    def evaluate(self) -> Any:
        raise NotImplementedError

    def export(self) -> Any:
        raise NotImplementedError

    def info(self) -> None:
        raise NotImplementedError

    def load(self) -> None:
        raise NotImplementedError

    def predict(self) -> Annotations:
        raise NotImplementedError

    def to_dataset(self) -> None:
        raise NotImplementedError

    def train(self) -> Any:
        raise NotImplementedError

    def validate(self) -> Any:
        raise NotImplementedError
