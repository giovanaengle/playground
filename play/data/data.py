from dataclasses import dataclass
from pathlib import Path

from .components.annotation import Annotations
from .components.image import Image
from .components.text import Text


@dataclass
class Data:
    name: str

    annotations: Annotations | None = None
    image: Image | None = None
    text: Text | None = None

    def copy(self) -> 'Data':
        return Data(
            annotations=self.annotations.copy(),
            image=self.image.copy(),
            name=self.name,
        )
    
    def load(self) -> None:
        self.annotations.load()
        self.image.load()

    def move(self, dst: Path) -> None:
        if self.annotations:
            self.annotations.parent = dst
        if self.image:
            self.image.parent = dst
        if self.text:
            self.text.parent = dst

    def save(self) -> None:
        if self.annotations:
            self.annotations.save(self.name)
        if self.image:
            self.image.save(self.name)
        if self.text:
            self.text.save(self.name)
            