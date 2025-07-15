from typing import Any

from ultralytics import settings, FastSAM, NAS, RTDETR, SAM, YOLO, YOLOWorld
from ultralytics.engine.results import Results
from ultralytics.utils import metrics

from common import Config
from core import TaskType
from data import Annotation, Bbox, Points2D
from .model import Model


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

        if self.name in networks:
            return networks[self.name]       
        else:
            raise Exception(f'Model not implemented: {self.name}')
    
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

    def info(self) -> None:
        if self.model:
            print(f'Architecture {self.name} technical information')
            self.model.info()
            print('\n')
        else:
            raise FileNotFoundError(f'Model not found')

    def load(self) -> None:
        architecture = self._create()
        print(f'Loading model architecture {self.name} \n')

        if self.weights:
            self.model = architecture(self.input).load(self.weights)
        else:
            self.model = architecture(self.input)

    def predict(self, data: Any) -> Results:
        config = self.config.sub('prediction')
        output = self.config.str('output')
        project = f'{output}/predict'
        
        predict_config = Config(path=config.str('config'))
        kwargs = predict_config.dict('params')

        results = self.model(source=data, project=project, **kwargs)
        return results
    
    def to_annotations(self, results: Results) -> list[Annotation]:
        if self.task == TaskType.CLASSIFY:
            config = self.config.sub('prediction')
            top5 = config.bool('top5')
            return ULPrediction.from_classify(results, top5)
        if self.task == TaskType.DETECT:
            return ULPrediction.from_detect(results)
        if self.task == TaskType.POSE:
            return ULPrediction.from_pose(results)
        if self.task == TaskType.SEGMENT:
            return ULPrediction.from_segment(results)

    def train(self) -> metrics:
        config = self.config.sub('train')
        data = config.str('data')
        project = self.config.str('output')
        
        train_config = Config(path=config.str('config'))
        kwargs = train_config.dict('params')

        results = self.model.train(data=data, project=project, **kwargs)
        print(results)

    def validate(self) -> metrics:
        config = self.config.sub('validation')
        data = config.str('data')
        project = self.config.str('output')
        
        val_config = Config(path=config.str('config'))
        kwargs = val_config.dict('params')

        metrics = self.model.val(data=data, project=project, **kwargs)
        print(metrics)

class ULPrediction:
    @staticmethod
    def from_classify(results: Results, top5: bool = False) -> list[Annotation]:
        if not results.probs:
            return
        
        items = []
 
        classes = results.names

        if top5:
            class_ids = results.probs.top5
            confidences = results.probs.top5conf.tolist()
            for index, class_id in enumerate(class_ids):
                anno = Annotation(
                    class_id = class_id,
                    class_name = classes[class_id],
                    confidence = round(confidences[index], 4),
                )
                items.append(anno)
        else:
            class_id = results.probs.top1
            confidence = results.probs.top1conf.tolist()
            class_name = classes[class_id]

            anno = Annotation(
                class_id = class_id,
                class_name = class_name,
                confidence = round(confidence, 4),
            )
            items.append(anno)
        
        return items
    
    @staticmethod
    def from_detect(results: Results) -> list[Annotation]:
        items = []

        classes = results.names
        
        if results.obb:
            for box in results.obb:
                bbox = box.xyxy.tolist()[0]
                bbox = Bbox(coords=bbox[:5], orientation=bbox[5])
                class_id = int(box.cls.tolist()[0])
                anno = Annotation(
                    bbox = bbox,
                    class_id = class_id,
                    class_name = classes[class_id],
                    confidence = round(box.conf.tolist()[0], 4),
                )
                items.append(anno)

        elif results.boxes:
            for box in results.boxes:
                bbox = box.xyxy.tolist()[0]
                bbox = Bbox(coords=bbox)
                class_id = int(box.cls.tolist()[0])

                anno = Annotation(
                    bbox = bbox,
                    class_id = class_id,
                    class_name = classes[class_id],
                    confidence = round(box.conf.tolist()[0], 4),
                )
                items.append(anno)

        else:
            return
         
        return items

    @staticmethod
    def from_pose(results: Results) -> list[Annotation]:
        if not results.keypoints:
            return
        
        boxes: list[Annotation] = ULPrediction.from_detect(results)
        if not items:
            return
        
        items = []
        for anno in boxes:
            points = results.keypoints.xy.tolist()[0]
            anno.points = Points2D(coords=points)
            items.append(anno)
        
        return items
        
    @staticmethod
    def from_segment(results: Results) -> list[Annotation]:
        if not results.masks:
            return
        
        boxes: list[Annotation] = ULPrediction.from_detect(results)
        if not boxes:
            return
        
        items = []
        for anno in boxes:
            points = results.masks.xy.tolist()[0]
            anno.points = Points2D(coords=points)
            items.append(anno)

        return items