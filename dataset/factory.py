from common import Config
from .dataset import Dataset
from .hf import HFDataset
from .ul import ULDataset


class DatasetFactory:
    datasets: dict[str, Dataset] = {
        'hugging_face': HFDataset,
        'ultralytics': ULDataset,
    }

    @staticmethod
    def create(config: Config) -> Dataset:
        name = config.str('framework')
        if name in DatasetFactory.datasets:
            return DatasetFactory.datasets[name](config)
        else:
            raise Exception(f'Dataset framework not implemented: {name}')