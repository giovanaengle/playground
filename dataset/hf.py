from dataclasses import dataclass

from data import Data
from .dataset import Dataset


@dataclass
class HFDataset(Dataset):
    def get(self) -> None:
        raise NotImplementedError

    def prepare(self, data: list[Data]) -> None:
        super().prepare(data)

    def save(self) -> None:
        raise NotImplementedError

    def setup(self) -> None:
        super().setup()
