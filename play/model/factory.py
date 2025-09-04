from play.common import Config

from .model import Model
from .ul import ULModel


class ModelFactory:
    models: dict[str, Model] = {
        'ultralytics': ULModel,
    }

    @staticmethod
    def create(config: Config) -> Model:
        name = config.str('framework')
        if name in ModelFactory.models:
            return ModelFactory.models[name](config)
        else:
            raise Exception(f'Model framework not implemented: {name}')
        