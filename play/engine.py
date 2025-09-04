from abc import ABC, abstractmethod
from datetime import datetime

from tqdm import tqdm

from .common import Config, Context
from .data import Annotations, Data
from .dataset import Dataset, DatasetFactory
from .model import Model, ModelFactory
from .data import (
    Ingestor, 
    IngestorFactory, 
    Job, 
    ProcessFactory, 
    Processor, 
    Storage, 
    StorageFactory
)


# === Abstracts ===

class Engine(ABC):
    def __init__(self, context: Context):
        self.context = context.sub('engine')
        self.context.logger.info(f'Engine initializing...')

    @abstractmethod
    def run(self) -> None:
        pass

class DataEngine(Engine):
    def __init__(self, context: Context):
        super().__init__(context)
        self.ingestor: Ingestor = IngestorFactory.create(self.context)
        self.storage: Storage = StorageFactory.create(self.context)

class ModelEngine(Engine):
    def __init__(self, context: Context):
        super().__init__(context)
        self.model: Model = ModelFactory.create(self.context)

# === Data Engines ===

class DatasetEngine(DataEngine):
    def __init__(self, context: Context):
        super().__init__(context)
        self.dataset: Dataset = DatasetFactory.create(self.config)
        
    def run(self) -> None:
        self.context.logger.info(f'Loading data for dataset...')
        with tqdm(total=self.ingestor.size()) as pbar:
            for data in self.ingestor.load():
                try:
                    self.context.logger.debug(f'Loading annotations for: {data.name}')
                    data.annotations.load()
                    self.context.logger.debug(f'Adding data to storage for: {data.name}')
                    self.storage.add(data)
                except Exception as e:
                    self.context.logger.error(f'Failed to add data {data.name}: {e}')

                pbar.set_description(f'{data.name}')
                pbar.update(1)

        self.context.logger.info(f'Preparing dataset...')
        self.dataset.setup()
        self.dataset.prepare(self.storage.all())
        self.dataset.save()
        self.context.logger.info(f'Dataset successfully created.')

class IngestEngine(DataEngine):
    def __init__(self, context: Context, batch_size: int = 100) -> None:
        super().__init__(context)
        self.batch_size = batch_size
        self.process: Processor = ProcessFactory.create(context)
        self.context.logger.info(f'Engine initialization completed.', engine='ingest')

    def _flush(self, batch: list[Data]) -> int:
        '''Persist a batch of data to storage.'''
        try:
            self.context.logger.debug('Flushing items to storage', size=len(batch))
            self.storage.add(batch)
            self.storage.save()
            self.storage.clear()
            return len(batch)
        except Exception:
            self.context.logger.exception(f'Failed to flush batch to storage')
            return 0
        
    def _handle_data(self, data: Data) -> Data:
        '''Load, process, and return data ready for storage. None on failure.'''
        try:
            self.context.logger.debug(f'Loading data for: {data.name}')
            data.load()
            self.context.logger.debug(f'Processing data for: {data.name}')
            job: Job = self.process.process(data=data)    

            return job.current
        except Exception:
            self.context.logger.exception(f'Failed to add data {data.name}')
            return None
        
    def run(self) -> None:
        self.context.logger.info('Running ingestion process.')
        persisted = 0
        seen = 0
        total = self.ingestor.size()
        with tqdm(total=total, desc='Ingesting') as pbar:
            batch: list[Data] = []  
            for data in self.ingestor.load():
                item = self._handle_data(data)

                if not item:
                    pbar.update(1)
                    continue

                batch.append(item)
                seen += 1

                if len(batch) >= self.batch_size:
                    persisted += self._flush(batch)
                    batch.clear()

                pbar.set_description(f'{data.name}')
                pbar.update(1)
        
        if batch:
            persisted += self._flush(batch)

        self.context.logger.info(f'Data successfully ingested. Processed {seen} items, persisted {persisted}/{total} items.')

# === Model Engines ===

class EvaluateEngine(ModelEngine):
    def run(self) -> None:
        self.context.logger.info('Evaluating model...')
        try:
            self.model.evaluate()
            self.context.logger.info('Model evaluation completed successfully.')
        except Exception as e:
            self.context.logger.error(f'Model evaluation failed: {e}')

class ExportEngine(ModelEngine):
    def run(self) -> None:
        self.context.logger.info('Exporting model...')
        try:
            self.model.export()
            self.context.logger.info('Model exported successfully.')
        except Exception as e:
            self.context.logger.error(f'Export failed: {e}')

class PredictEngine(ModelEngine):
    def __init__(self, context: Context):
        super().__init__(context)
        data_config: Config = self.context.config
        self.ingestor: Ingestor = IngestorFactory.create(data_config)

    def run(self) -> None:
        dst_path = self.config.path('output').joinpath('predict')
        dst_path.mkdir(parents=True, exist_ok=True)
        self.context.logger.info(f'Prediction output directory created: {dst_path}')

        ts = datetime.now().replace(microsecond=0).isoformat()
        results_path = dst_path.joinpath(f'{ts}.csv')

        self.context.logger.info(f'Saving predictions to: {results_path}')
        with open(results_path, 'a',newline='') as file:
            file.write('image,class_name,class_score\n')

            self.context.logger.info(f'Starting predictions...')
            with tqdm(total=self.ingestor.size()) as pbar:
                for data in self.ingestor.load():
                    try:
                        self.context.logger.debug(f'Loading image for: {data.name}')
                        data.image.load()
                        results = self.model.predict(data.image.content)
                        if results:
                            self.context.logger.debug(f'Predictions found for {data.name}, converting to annotations')
                            predictions = self.model.to_annotations(results[0]) 

                            data.annotations = Annotations(
                                items=predictions,
                                name=data.name,
                                parent=dst_path,
                                suffix=''
                            )
                            data.annotations.save(data.name)

                            self.context.logger.debug(f'Writing predictions for {data.name} to csv file')
                            for anno in data.annotations.items:
                                file.write(f'{data.name},{anno.class_name},{anno.confidence}\n')
                        else:
                            self.context.logger.warning(f'No predictions found for {data.name}')
                    except Exception as e:
                        self.context.logger.error(f'Failed to predict for {data.name}: {e}')

                    pbar.set_description(f'{data.name}')
                    pbar.update(1)

        self.context.logger.info('Predictions successfully concluded.')
                
class TrainEngine(ModelEngine):
    def run(self) -> None:
        self.context.logger.info('Starting model training...')
        try:
            results = self.model.train()
            self.context.logger.info('Model training completed successfully.')
            # TODO: add results to report
        except Exception as e:
            self.context.logger.error(f'Model training failed: {e}')

class ValidateEngine(ModelEngine):
    def run(self) -> None:
        self.context.logger.info('Starting model validation...')
        try:
            results = self.model.validate()
            self.context.logger.info('Model validation completed successfully.')
            # TODO: add results to report
        except Exception as e:
            self.context.logger.error(f'Model validation failed: {e}')

# === Factory ===

class EngineFactory:
    @staticmethod
    def create(context: Context, engine: str) -> Engine:
        engine_context = context.sub(engine)
        match engine:
            case 'dataset':
                return DatasetEngine(engine_context)
            case 'evaluate':
                return EvaluateEngine(engine_context)
            case 'export':
                return ExportEngine(engine_context)
            case 'ingest':
                return IngestEngine(engine_context)
            case 'predict':
                return PredictEngine(engine_context)
            case 'train':
                return TrainEngine(engine_context)
            case 'validate':
                return ValidateEngine(engine_context)
            case _:
                raise ValueError(f'Unknown engine type: {engine}')
