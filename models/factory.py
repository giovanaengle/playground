from typing import Any

from common import Config
from .model import Model, ULModel


class ModelFactory:
    models: dict[str, Any] = {
        'ultralytics': ULModel,
    }

    @staticmethod
    def create(config: Config) -> Model:
        name = config.str('framework')
        if name in ModelFactory.models:
            return ModelFactory.models[name](config)
        else:
            raise Exception(f'Model not implemented: {name}')
        