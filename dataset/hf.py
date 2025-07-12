from dataclasses import dataclass

from common import Config
from data import Data
from .dataset import Dataset


@dataclass
class HFDataset(Dataset):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def get(self) -> None:
        return super().setup()
    
    def prepare(self, data: list[Data]) -> None:
        super().prepare(data)

    def save(self) -> None:
        return super().setup()

    def setup(self) -> None:
        return super().setup()