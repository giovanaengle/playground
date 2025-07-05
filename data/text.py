from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .media import Media


@dataclass
class Text(Media):
    name: str
    suffix: str | None
    
    content: dict[str,str] | None = None
    parent: Path | None = None

    def copy(self) -> None:
        NotImplemented
    
    def load(self) -> None:
        NotImplemented
    
    def save(self) -> None:
        NotImplemented