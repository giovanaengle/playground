from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import ParseResult, urlparse
import urllib.request


@dataclass
class Media(ABC):
    name: str
    parent: Path
    suffix: str

    @abstractmethod
    def copy(self) -> None:
        NotImplemented
        
    def load(self, cache: Path = Path('.experiments/cache')) -> None:
        path = self.parent.joinpath(f'{self.name}{self.suffix}')
        
        if str(path).startswith('http'):
            dst = cache.joinpath(self.name, self.suffix)
            urllib.request.urlretrieve(path, dst)
            self.parent = cache
        
        elif not path.exists():
            raise Exception(f'Data not found in {path}')

    @abstractmethod
    def save(self) -> None:
        NotImplemented
