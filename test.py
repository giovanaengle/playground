import argparse

from tqdm import tqdm

from common import Config
from core.storage import Storage, StorageFactory
from data import Job, ProcessFactory, Processor
from dataset import DatasetFactory
from core import InputFactory
from models import ModelFactory


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    # Config 
    config = Config(path=args.config_path)
    
    data_config = config.sub('data')
    dataset_config = config.sub('dataset')
    model_config = config.sub('model')

    # Data
    storage: Storage = StorageFactory.create(config)

    ## input and pre-process
    input = InputFactory.create(data_config)
    processes: Processor = ProcessFactory.create(data_config)
    storage.clear()

    with tqdm(total=input.size()) as pbar:
        for data in input.load():
            data.load()
            job: Job = processes.process(data=data)

            storage.add(data=job.current)
            storage.save()
            storage.clear()

            pbar.set_description(f'{data.name}')
            pbar.update(1)

    ## send to dataset
    input = InputFactory.create(dataset_config)
    dataset = DatasetFactory.create(dataset_config)

    with tqdm(total=input.size()) as pbar:
        for data in input.load():
            data.annotations.load()
            storage.add(data)

            pbar.set_description(f'{data.name}')
            pbar.update(1)
    
    dataset.setup()
    dataset.prepare(storage.all())
    dataset.save()
    
    # Model
    model_engine = ModelFactory.create(model_config)



