from abc import ABC, abstractmethod
from pathlib import Path
from shutil import rmtree

from play.common import Context
from play.data import Data


class Storage(ABC):
    def __init__(self, context: Context):
        self.context = context
        self.items: dict[str, Data] = {}
        parent = context.config.path('parent')
        project = context.config.str('project')
        self.path = parent.joinpath(project, 'storage')

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
    def __init__(self, context: Context) -> None:
        super().__init__(context)

    def add(self, data: Data | list[Data]) -> None:
        if isinstance(data, Data):
            data = [data]
        elif isinstance(data, list) and data and isinstance(data[0], list):
            data = [d for sublist in data for d in sublist]
        
        self.items.update({d.name: d for d in data})

    def all(self) -> list[Data]:
        return list(self.items.values())

    def clear(self) -> list[Data]:
        cleared = list(self.items.values())
        self.items.clear()
        return cleared

    def get(self, name: str) -> Data | None:
        return self.items.get(name)

    def save(self) -> None:
        dirs = {
            'annotations': self.path.joinpath('annotations'),
            'images': self.path.joinpath('images'),
            'texts': self.path.joinpath('texts'),
        }
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        for data in self.items.values():
            if data.annotations:
                data.annotations.parent = self.path.joinpath('annotations')
            if data.image:
                data.image.parent = self.path.joinpath('images')
            if data.text:
                data.text.parent = self.path.joinpath('texts')  
            data.save()

    def set(self, name: str, data: Data) -> None:
        self.items[name] = data

    def setup(self):
        return super().setup()
    
    def unset(self, name: str) -> None:
        self.items.pop(name, None)

class StorageFactory:
    @staticmethod
    def create(context: Context) -> Storage:
        storage_ctx = context.sub('storage')
        return LocalStorage(storage_ctx)