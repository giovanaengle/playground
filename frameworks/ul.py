from dataclasses import dataclass

from ultralytics import settings
from ultralytics.engine.results import Results
from ultralytics.utils import metrics

from data import Annotation, Annotations, Bbox, Points2D
from .framework import Framework


@dataclass
class UlFramework(Framework):
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

    def _set_sys(self) -> None:
        kwargs = self.config.dict('settings')
        settings.update(**kwargs)
        print(f'Settings: {settings}')

    async def predict(self) -> Annotations:
        kwargs = self.config.dict('predict')
        results = self.model(source = self.data, **kwargs)
        return self._to_annotation(results[0])

    def to_dataset(self):
        return super().to_dataset()
    
    def train(self) -> metrics:
        kwargs = self.config.dict('train')
        results = self.model.train(**kwargs)
        return results

    def validate(self) -> metrics:
        kwargs = self.config.dict('validate')
        metrics = self.model.val(**kwargs)
        return metrics
