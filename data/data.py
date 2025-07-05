from dataclasses import dataclass
from pathlib import Path

from .annotation import Annotations
from .image import Image
from .text import Text


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
        self.annotations.parent = dst
        self.image.parent = dst
        self.text.parent = dst

    def save(self) -> None:
        self.annotations.save()
        self.image.save()