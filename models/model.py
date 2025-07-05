from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ultralytics import FastSAM, NAS, RTDETR, SAM, YOLO, YOLOWorld

from common import Config


class Model(ABC):
    classes: dict[int, str] = field(default_factory=lambda: {})
    model: Any | None = None

    def __init__(self, config: Config) -> None:
        self.config = config

        self.name = config.str('architecture')
        self.weights = config.str('weights')
        self.path = Path(config.str('path'))
        if not self.path.exists():
            raise Exception(f'Model path does not exist: {self.path}')

        self._set()

    def _set(self) -> None:
        self.load()
        self.categories()
        self.info() 

    @abstractmethod
    def categories(self) -> None:
        NotImplemented

    @abstractmethod
    def info(self) -> None:
        NotImplemented

    @abstractmethod
    def load(self) -> None:
        NotImplemented

class ULModel(Model):
    def __init__(self, config: Config):
        super().__init__(config)
    
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
    
    def _set(self) -> None:
        super()._set()

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
