from typing import Any

from ultralytics import settings, FastSAM, NAS, RTDETR, SAM, YOLO, YOLOWorld
from ultralytics.engine.results import Results
from ultralytics.utils import metrics

from common import Config
from data import Annotation, Annotations, Bbox, Media, Points2D
from model import Model


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

    def _to_annotation(self, results: Results) -> Annotations:
        if not results:
            return 
        
        bbox = None
        points = None

        # check and collect for classification results
        if results.probs:
            classes = results.names
            class_id = results.probs.top1
            class_name = classes[class_id]
            confidence = results.probs.top1conf.tolist()
        
        # check and collect for detection results
        elif results.boxes:
            bbox = results.boxes.xyxy.tolist()[0]
            class_id = results.boxes.id.tolist()[0]
            class_name = results.boxes.cls.tolist()[0]
            confidence = results.boxes.conf.tolist()[0]

        # check and collect for oriented detection results
        elif results.obb:
            bbox = results.obb.xyxy.tolist()[0]
            bbox = Bbox(coords=bbox[:5], orientation=bbox[5])
            class_id = results.obb.id.tolist()[0]
            class_name = results.obb.cls.tolist()[0]
            confidence = results.obb.conf.tolist()[0]

        # check and collect for segmentation results
        if results.masks:
            points = results.masks.xy.tolist()[0]

        # check and collect for keypoints results
        elif results.keypoints:
            points = results.keypoints.xy.tolist()[0]

        confidence = round(confidence, 4)
        if points:
            points = Points2D(coords=points)

        annotation = Annotation(
            bbox=bbox,
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            points=points
        )

        return annotation

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
            self.model = architecture(self.path).load(self.weights)
        else:
            self.model = architecture(self.path)

    async def predict(self, data: Media) -> Annotations:
        kwargs = self.config.dict('predict')
        results = self.model(source=data, **kwargs)
        return self._to_annotation(results[0])
    
    def train(self) -> metrics:
        kwargs = self.config.dict('train')
        results = self.model.train(**kwargs)
        return results

    def validate(self) -> metrics:
        kwargs = self.config.dict('validate')
        metrics = self.model.val(**kwargs)
        return metrics