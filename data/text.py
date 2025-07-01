from dataclasses import dataclass

import numpy as np

from .media import Media


@dataclass
class Text(Media):
    content: np.ndarray | None = None
    path: str | None = None

    def copy(self) -> None:
        NotImplemented
    
    def load(self) -> None:
        NotImplemented
    
    def save(self) -> None:
        NotImplemented