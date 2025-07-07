from dataclasses import dataclass
from typing import Any

from data import Annotations
from .framework import Framework


@dataclass
class HfFramework(Framework):
    def _to_annotation(self) -> Annotations:
        NotImplemented 

    def predict(self) -> Annotations:
        NotImplemented

    def to_dataset(self) -> None:
        NotImplemented

    def train(self) -> Any:
        NotImplemented

    def validate(self) -> Any:
        NotImplemented
