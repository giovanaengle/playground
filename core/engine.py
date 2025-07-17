
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from common import Config
from data import Job, ProcessFactory, Processor
from dataset import Dataset, DatasetFactory
from .input import Input, InputFactory
from models import Model, ModelFactory
from .storage import Storage, StorageFactory


# Abstracts

@dataclass
class Engine(ABC):
    config: Config

    @abstractmethod
    def run(self) -> None:
        NotImplemented

@dataclass
class DataEngine(Engine):
    input: Input
    storage: Storage

@dataclass
class ModelEngine(Engine):
    model: Model

# Engines

@dataclass
class EvaluateEngine(ModelEngine):
    def run(self) -> None:
        self.model.evaluate()

@dataclass
class ExportEngine(ModelEngine):
    def run(self) -> None:
        self.model.export()

@dataclass
class DatasetEngine(DataEngine):
    dataset: Dataset

    def run(self) -> None:
        with tqdm(total=self.input.size()) as pbar:
            for data in self.input.load():
                data.annotations.load()
                self.storage.add(data)

                pbar.set_description(f'{data.name}')
                pbar.update(1)

        self.dataset.setup()
        self.dataset.prepare(self.storage.all())
        self.dataset.save()

@dataclass
class IngestEngine(DataEngine):
    process: Processor

    def run(self) -> None:
        with tqdm(total=self.input.size()) as pbar:
            for data in self.input.load():
                data.load()

                job: Job = self.process.process(data=data)
                
                self.storage.add(data=job.current)
                self.storage.save()
                self.storage.clear()

                pbar.set_description(f'{data.name}')
                pbar.update(1)

@dataclass
class PredictEngine(DataEngine):
    model: Model

    def run(self) -> None:
        config = self.config.sub('model')
        dst_path = config.path('output').joinpath('predict')
        dst_path.mkdir(parents=True, exist_ok=True)
        
        with tqdm(total=self.input.size()) as pbar:
            for data in self.input.load():
                data.image.load()
                results = self.model.predict(data.image.content)
                if results:
                    data.image = None
                    data.annotations.items = self.model.to_annotations(results[0]) # only one image per batch
                    data.annotations.parent = dst_path
                    data.annotations.save(data.name)
                    self.storage.add(data)

                pbar.set_description(f'{data.name}')
                pbar.update(1)

        ts: str = datetime.now().replace(microsecond=0).isoformat()
        results_path = dst_path.joinpath(f'{ts}.csv')
        
        print(f'Preparing csv file saved at {dst_path}')
        with open(results_path, 'w') as file:
            file.write('image,class_name,class_score\n')
            for data in self.storage.all():
                for anno in data.annotations.items:
                    file.write(f'{data.name},{anno.class_name},{anno.confidence}\n')

@dataclass
class TrainEngine(ModelEngine):
    def run(self) -> None:
        self.model.train()

@dataclass
class ValidateEngine(ModelEngine):
    def run(self) -> None:
        self.model.validate()

class EngineFactory:
    @staticmethod
    def create(config: Config, engine: str) -> Engine:
        if engine == 'dataset':
            return DatasetEngine(
                config=config,
                dataset=DatasetFactory.create(config=config.sub('dataset')),
                input=InputFactory.create(config=config.sub('dataset')),
                storage=StorageFactory.create(config=config.sub('dataset')),
            )
        elif engine == 'evaluate':
            return EvaluateEngine(
                config=config,
                model=ModelFactory.create(config=config.sub('model')),
            )
        elif engine == 'export':
            return ExportEngine(
                config=config,
                model=ModelFactory.create(config=config.sub('model')),
            )
        elif engine == 'ingest':
            return IngestEngine(
                config=config,
                input=InputFactory.create(config=config.sub('data')),
                process=ProcessFactory.create(config=config.sub('data')),
                storage=StorageFactory.create(config=config.sub('data')),
            )
        elif engine == 'predict':
            return PredictEngine(
                config=config,
                input=InputFactory.create(config=config.sub('dataset')),
                model=ModelFactory.create(config=config.sub('model')),
                storage=StorageFactory.create(config=config.sub('model')),
            )
        elif engine == 'train':
            return TrainEngine(
                config=config,
                model=ModelFactory.create(config=config.sub('model')),
            )
        elif engine == 'validate':
            return ValidateEngine(
                config=config,
                model=ModelFactory.create(config=config.sub('model')),
            )
        else:
            raise Exception(f'unknown engine type: {engine}')