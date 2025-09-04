from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .component import Component


@dataclass
class Image(Component):
    content: np.ndarray = field(default_factory=lambda: np.empty((0)))

    def copy(self) -> None:
        if self.is_empty():
            self.load()

        return Image(
            content=self.content.copy() if not self.is_empty() else np.empty((0)),
            name=self.name,
            parent=self.parent,
            suffix=self.suffix,
        )

    def is_empty(self) -> bool:
        return self.content.size <= 0
    
    def load(self) -> None:
        if not self.parent or not self.is_empty():
            return
        
        path = self.parent.joinpath(f'{self.name}{self.suffix}')
        if not path.exists():
            return 
        self.content = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    def path(self) -> Path:
        return self.parent.joinpath(f'{self.name}{self.suffix}')
    
    def save(self, name: str) -> None:
        if not self.is_empty():
            path = self.parent.joinpath(f'{name}{self.suffix}')
            cv2.imwrite(path, self.content)
