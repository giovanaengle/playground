from dataclasses import dataclass
from typing import Any

from data.annotation import Annotation
from .framework import Framework


@dataclass
class HfFramework(Framework):
    def _to_annotation(self) -> Annotation:
        NotImplemented 

    def predict(self) -> Annotation:
        NotImplemented

    def train(self) -> Any:
        NotImplemented

    def validate(self) -> Any:
        NotImplemented
