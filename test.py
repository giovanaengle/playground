import argparse

from common import Config
from data import Data, Job, ProcessFactory, Processor
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
    framework_config = config.sub('framework')

    # Data
    input = InputFactory.create(data_config)
    processes: Processor = ProcessFactory.create(data_config)
    
    total= input.size()
    for data in input.load():
        data.load()

        job: Job = processes.process(data=data)
        data: Data = job.current[0]
        exit()
    
    # Model
    model_engine = ModelFactory.create(model_config)



