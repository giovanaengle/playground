from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Component(ABC):
    '''Base class for all data components.'''
    name: str
    parent: Path
    suffix: str

    @abstractmethod
    def copy(self) -> None:
        raise NotImplementedError
        
    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError
