from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from common import Config
from data import Data


@dataclass
class Storage(ABC):
    path: Path

    @abstractmethod
    def all(self) -> list[Data]:
        NotImplemented

    @abstractmethod
    def add(self, data: Data | list[Data]) -> None:
        NotImplemented

    @abstractmethod
    def get(self, name: str) -> Data:
        NotImplemented

    @abstractmethod
    def save(self) -> None:
        NotImplemented

    @abstractmethod
    def set(self, name: str, data: Data) -> None:
        NotImplemented

    @abstractmethod
    def unset(self, name: str) -> None:
        NotImplemented

@dataclass
class LocalStorage(Storage):
    items: dict[str, Data] = field(default_factory=lambda: {})

    def add(self, data: Data | list[Data]) -> None:
        if type(data) == Data:
            data = [data]

        for data in data:
            self.items[data.name] = data

    def all(self) -> list[Data]:
        return list(self.items.values())

    def clear(self) -> list[Data]:
        self.items = {}
        
    def get(self, name: str) -> Data | None:
        return self.items[name]

    def save(self) -> None:
        self.path.mkdir(exist_ok=True, parents=True)

        for data in self.items.values():
            data.move(self.path)
            data.save()

    def set(self, name: str, data: Data) -> None:
        self.items[name] = data

    def unset(self, name: str) -> None:
        del self.items[name]

class StorageFactory:
    @staticmethod
    def create(config: Config) -> Storage:
        return LocalStorage(path=config.path('storage'))