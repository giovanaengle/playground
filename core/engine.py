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
        self.logger.info(f'Loading data for dataset...')
        with tqdm(total=self.input.size()) as pbar:
            for data in self.input.load():
                try:
                    self.logger.debug(f'Loading annotations for: {data.name}')
                    data.annotations.load()
                    self.logger.debug(f'Adding data to storage for: {data.name}')
                    self.storage.add(data)
                except Exception as e:
                    self.logger.error(f'Failed to add data {data.name}: {e}')

                pbar.set_description(f'{data.name}')
                pbar.update(1)

        self.logger.info(f'Preparing dataset...')
        self.dataset.setup()
        self.dataset.prepare(self.storage.all())
        self.dataset.save()
        self.logger.info(f'Dataset successfully created.')

class IngestEngine(DataEngine):
    def __init__(self, context: Context):
        super().__init__(context)
        self.process: Processor = ProcessFactory.create(self.config.sub('process'))

    def run(self) -> None:
        self.logger.info(f'Ingesting data...')
        with tqdm(total=self.input.size()) as pbar:
            for data in self.input.load():
                try:
                    self.logger.debug(f'Loading data for: {data.name}')
                    data.load()
                    self.logger.debug(f'Processing data for: {data.name}')
                    job: Job = self.process.process(data=data)    
                    self.logger.debug(f'Adding data to storage for: {data.name}')
                    self.storage.add(data=job.current)
                    self.storage.save()
                    self.storage.clear()

                except Exception as e:
                    self.logger.error(f'Failed to add data {data.name}: {e}')

                pbar.set_description(f'{data.name}')
                pbar.update(1)
        
        self.logger.info(f'Data successfully added.')

# === Model Engines ===

class EvaluateEngine(ModelEngine):
    def run(self) -> None:
        self.logger.info('Evaluating model...')
        try:
            self.model.evaluate()
            self.logger.info('Model evaluation completed successfully.')
        except Exception as e:
            self.logger.error(f'Model evaluation failed: {e}')

class ExportEngine(ModelEngine):
    def run(self) -> None:
        self.logger.info('Exporting model...')
        try:
            self.model.export()
            self.logger.info('Model exported successfully.')
        except Exception as e:
            self.logger.error(f'Export failed: {e}')

class PredictEngine(ModelEngine):
    def __init__(self, context: Context):
        super().__init__(context)
        data_config: Config = Config(path=self.config.path('data'))
        self.input: Input = InputFactory.create(data_config)

    def run(self) -> None:
        dst_path = self.config.path('output').joinpath('predict')
        dst_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f'Prediction output directory created: {dst_path}')

        ts = datetime.now().replace(microsecond=0).isoformat()
        results_path = dst_path.joinpath(f'{ts}.csv')

        self.logger.info(f'Saving predictions to: {results_path}')
        with open(results_path, 'a',newline='') as file:
            file.write('image,class_name,class_score\n')

            self.logger.info(f'Starting predictions...')
            with tqdm(total=self.input.size()) as pbar:
                for data in self.input.load():
                    try:
                        self.logger.debug(f'Loading image for: {data.name}')
                        data.image.load()
                        results = self.model.predict(data.image.content)
                        if results:
                            self.logger.debug(f'Predictions found for {data.name}, converting to annotations')
                            predictions = self.model.to_annotations(results[0]) 

                            data.annotations = Annotations(
                                items=predictions,
                                name=data.name,
                                parent=dst_path,
                                suffix=''
                            )
                            data.annotations.save(data.name)

                            self.logger.debug(f'Writing predictions for {data.name} to csv file')
                            for anno in data.annotations.items:
                                file.write(f'{data.name},{anno.class_name},{anno.confidence}\n')
                        else:
                            self.logger.warning(f'No predictions found for {data.name}')
                    except Exception as e:
                        self.logger.error(f'Failed to predict for {data.name}: {e}')

                    pbar.set_description(f'{data.name}')
                    pbar.update(1)

        self.logger.info('Predictions successfully concluded.')
                
class TrainEngine(ModelEngine):
    def run(self) -> None:
        self.logger.info('Starting model training...')
        try:
            results = self.model.train()
            self.logger.info('Model training completed successfully.')
            # TODO: add results to report
        except Exception as e:
            self.logger.error(f'Model training failed: {e}')

class ValidateEngine(ModelEngine):
    def run(self) -> None:
        self.logger.info('Starting model validation...')
        try:
            results = self.model.validate()
            self.logger.info('Model validation completed successfully.')
            # TODO: add results to report
        except Exception as e:
            self.logger.error(f'Model validation failed: {e}')

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
