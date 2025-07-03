from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from common import Config
from data.annotation import Annotations
from data.data import Data
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

    def train(self) -> Any:
        NotImplemented

    def validate(self) -> Any:
        NotImplemented
