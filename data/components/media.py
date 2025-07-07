from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import ParseResult, urlparse
import requests


@dataclass
class Media(ABC):
    name: str
    parent: Path
    suffix: str

    @abstractmethod
    def copy(self) -> None:
        NotImplemented
        
    @abstractmethod
    def load(self) -> None:
        NotImplemented

    @abstractmethod
    def save(self) -> None:
        NotImplemented
