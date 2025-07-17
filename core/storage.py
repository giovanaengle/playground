from abc import ABC, abstractmethod
from pathlib import Path
from shutil import rmtree

from common import Config
from data import Data


class Storage(ABC):
    def __init__(self, path: Path):
        self.path = path

    @abstractmethod
    def add(self, data: Data | list[Data]) -> None:
        NotImplemented

    @abstractmethod
    def all(self) -> list[Data]:
        NotImplemented
    
    @abstractmethod
    def clear(self, data: Data | list[Data]) -> None:
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

    def setup(self):
        rmtree(self.path, ignore_errors=True, onexc=None)
        self.path.mkdir(exist_ok=True, parents=True)

    @abstractmethod
    def unset(self, name: str) -> None:
        NotImplemented


class LocalStorage(Storage):
    items: dict[str, Data] = {}

    def __init__(self, path: Path) -> None:
        super().__init__(path)

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
        for data in self.items.values():
            if data.annotations:
                data.annotations.parent = self.path.joinpath('annotations')
                data.annotations.parent.mkdir(parents=True, exist_ok=True)
            if data.image:
                data.image.parent = self.path.joinpath('images')
                data.image.parent.mkdir(parents=True, exist_ok=True)
            if data.text:
                data.text.parent = self.path.joinpath('texts')
                data.text.parent.mkdir(parents=True, exist_ok=True)

            data.save()

    def set(self, name: str, data: Data) -> None:
        self.items[name] = data

    def setup(self):
        return super().setup()
    
    def unset(self, name: str) -> None:
        del self.items[name]

class StorageFactory:
    @staticmethod
    def create(config: Config) -> Storage:
        return LocalStorage(path=config.path('output'))