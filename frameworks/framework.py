from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from common import Config
from data.components.annotation import Annotations
from data.components.data import Data
from models import Model


@dataclass
class Framework(ABC):
    config: Config

    data: Data | None = None
    model: Model | None = None
    
    @abstractmethod
    def _to_annotation(self) -> Annotations:
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
