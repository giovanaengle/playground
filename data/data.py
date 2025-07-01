from dataclasses import dataclass
from pathlib import Path

from .annotation import Annotations
from .image import Image
from .text import Text


@dataclass
class Data:
    name: str
    path: Path

    annotations: Annotations | None = None
    image: Image | None = None
    text: Text | None = None

    def copy(self) -> 'Data':
        return Data(
            annotations=self.annotations.copy(),
            image=self.image.copy(),
            name=self.name,
            path=Path(self.path.absolute()),
        )

    def load(self) -> None:
        self.annotations.load()
        self.image.load()

    def move(self, dst: Path) -> None:
        self.annotations.path = dst.joinpath(self.annotations.path.name)
        self.image.parent = dst
        self.path = dst.joinpath(self.path.name)

    def save(self) -> None:
        self.annotations.save()
        self.image.save()