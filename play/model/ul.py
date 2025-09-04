from typing import Any

from ultralytics import settings, FastSAM, NAS, RTDETR, SAM, YOLO, YOLOWorld
from ultralytics.engine.results import Results
from ultralytics.utils import metrics
from ultralytics.utils.benchmarks import benchmark

from play.common import Config, TaskType
from play.data import Annotation

from .model import Model
from .parsers.ul import ULParser


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

        if self.architecture in networks:
            return networks[self.architecture]       
        else:
            raise Exception(f'Model not implemented: {self.architecture}')
    
    def _set(self) -> None:
        super()._set()

    def _sys(self) -> None:
        kwargs = self.config.dict('settings')
        settings.update(**kwargs)
        print(f'Settings: {settings}')

    def categories(self) -> None:
        if self.model:
            self.classes = self.model.model.names
        else:
            raise FileNotFoundError(f'Model not found')

    def evaluate(self) -> None:
        kwargs = self.params.dict('evaluate')
        benchmark(self.model, data=self.data, **kwargs)

    def export(self) -> None:
        kwargs = self.params.dict('export')
        self.model.export(data=self.data, **kwargs)

    def info(self) -> None:
        if self.model:
            print(f'Architecture {self.architecture} technical information')
            self.model.info()
            print('\n')
        else:
            raise FileNotFoundError(f'Model not found')

    def load(self) -> None:
        model_interface = self._create()
        print(f'Loading model architecture {self.architecture} \n')

        if self.weights:
            self.model = model_interface(self.path).load(self.weights)
        else:
            self.model = model_interface(self.path)

    def predict(self, data: Any) -> Results:
        kwargs = self.params.dict('predict')
        results = self.model(source=data, project=f'{self.output}/predict', **kwargs)
        return results
    
    def to_annotations(self, results: Results) -> list[Annotation]:
        if self.task == TaskType.CLASSIFY:
            top5 = self.config.bool('top5')
            return ULParser.from_classify(results, top5)
        if self.task == TaskType.DETECT:
            return ULParser.from_detect(results)
        if self.task == TaskType.POSE:
            return ULParser.from_pose(results)
        if self.task == TaskType.SEGMENT:
            return ULParser.from_segment(results)

    def train(self) -> metrics:
        kwargs = self.params.dict('train')
        results = self.model.train(data=self.data, project=self.output, **kwargs)
        return results

    def validate(self) -> metrics:
        kwargs = self.params.dict('validate')
        results = self.model.val(data=self.data, project=self.output, **kwargs)
        return results
