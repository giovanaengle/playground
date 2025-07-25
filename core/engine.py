from abc import ABC, abstractmethod
from datetime import datetime

from tqdm import tqdm

from common import Config, Context, Logger, Metrics
from data import Annotations, Job, ProcessFactory, Processor
from dataset import Dataset, DatasetFactory
from .input import Input, InputFactory
from models import Model, ModelFactory
from .storage import Storage, StorageFactory


# === Abstracts ===

class Engine(ABC):
    def __init__(self, context: Context):
        self.context = context
        self.config: Config = context.config
        self.logger: Logger = context.logger
        self.metrics: Metrics = context.metrics

    @abstractmethod
    def run(self) -> None:
        pass

class DataEngine(Engine):
    def __init__(self, context: Context):
        super().__init__(context)
        self.input: Input = InputFactory.create(self.config)
        self.storage: Storage = StorageFactory.create(self.config)

class ModelEngine(Engine):
    def __init__(self, context: Context):
        super().__init__(context)
        self.model: Model = ModelFactory.create(self.config)

# === Data Engines ===

class DatasetEngine(DataEngine):
    def __init__(self, context: Context):
        super().__init__(context)
        self.dataset: Dataset = DatasetFactory.create(self.config)
        
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

class IngestEngine(DataEngine):
    def __init__(self, context: Context):
        super().__init__(context)
        self.process: Processor = ProcessFactory.create(self.config.sub('process'))

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

# === Model Engines ===

class EvaluateEngine(ModelEngine):
    def run(self) -> None:
        self.model.evaluate()

class ExportEngine(ModelEngine):
    def run(self) -> None:
        self.model.export()

class PredictEngine(ModelEngine):
    def __init__(self, context: Context):
        super().__init__(context)
        data_config: Config = Config(path=self.config.path('data'))
        self.input: Input = InputFactory.create(data_config)

    def run(self) -> None:
        dst_path = self.config.path('output').joinpath('predict')
        dst_path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().replace(microsecond=0).isoformat()
        results_path = dst_path.joinpath(f'{ts}.csv')

        self.logger.info(f'Saving predictions to: {results_path}')
        with open(results_path, 'a',newline='') as file:
            file.write('image,class_name,class_score\n')

            with tqdm(total=self.input.size()) as pbar:
                for data in self.input.load():
                    data.image.load()
                    results = self.model.predict(data.image.content)
                    if results:
                        predictions = self.model.to_annotations(results[0]) 

                        data.annotations = Annotations(
                            items=predictions,
                            name=data.name,
                            parent=dst_path,
                            suffix=''
                        )
                        data.annotations.save(data.name)

                        for anno in data.annotations.items:
                            file.write(f'{data.name},{anno.class_name},{anno.confidence}\n')


                    pbar.set_description(f'{data.name}')
                    pbar.update(1)

                
class TrainEngine(ModelEngine):
    def run(self) -> None:
        results = self.model.train()
        # TODO: add results to report

class ValidateEngine(ModelEngine):
    def run(self) -> None:
        results = self.model.validate()
        # TODO: add results to report

# === Factory ===

class EngineFactory:
    @staticmethod
    def create(context: Context, engine: str) -> Engine:
        match engine:
            case 'dataset':
                return DatasetEngine(context)
            case 'evaluate':
                return EvaluateEngine(context)
            case 'export':
                return ExportEngine(context)
            case 'ingest':
                return IngestEngine(context)
            case 'predict':
                return PredictEngine(context)
            case 'train':
                return TrainEngine(context)
            case 'validate':
                return ValidateEngine(context)
            case _:
                raise ValueError(f'Unknown engine type: {engine}')
