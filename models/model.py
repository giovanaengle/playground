from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ultralytics import FastSAM, NAS, RTDETR, SAM, YOLO, YOLOWorld

from common import Config


@dataclass
class Model(ABC):
    config: Config
    
    classes: dict[int, str] = field(default_factory=lambda: {})
    model: Any | None = None
    name: str | None = None
    path: Path | None = None
    weights: Any | None = None

    @abstractmethod
    def categories(self) -> None:
        NotImplemented

    @abstractmethod
    def info(self) -> None:
        NotImplemented

    @abstractmethod
    def load(self) -> None:
        NotImplemented

    def set(self) -> None:
        self.name = self.config.str('architecture')
        self.weights = self.config.str('weights')
        self.path = Path(self.config.str('path'))

        if not self.path.exists():
            raise Exception(f'Model path does not exist: {path}')

        self.load()
        self.categories()
        self.info()

class ULModel(Model):
    def _create(self) -> Any:
        networks: dict[str, Any] = {
            'fastsam': FastSAM,
            'nas': NAS,
            'rtdetr': RTDETR,
            'sam': SAM,
            'yolo': YOLO,
            'yoloworld': YOLOWorld,
        }

        if self.name in networks:
            return networks[self.name]       
        else:
            raise Exception(f'Model not implemented: {self.name}')

    def categories(self) -> None:
        if self.model:
            self.classes = self.model.model.names
        else:
            raise FileNotFoundError(f'Model not found')

    def info(self) -> None:
        if self.model:
            print(f'Architecture {self.name} technical information')
            self.model.info()
            print('\n')
        else:
            raise FileNotFoundError(f'Model not found')

    def load(self) -> None:
        architecture = self._create()
        print(f'Loading model architecture {self.name} \n')

        if self.weights:
            self.model = architecture(self.path).load(self.weights)
        else:
            self.model = architecture(self.path)
    
    def set(self) -> None:
        super().set()