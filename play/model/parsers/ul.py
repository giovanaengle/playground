from ultralytics.engine.results import Results

from play.data import Annotation, Bbox, Points2D


class ULParser:
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
                bbox.to_int()

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
                bbox.to_int()

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
        
        boxes: list[Annotation] = ULParser.from_detect(results)
        if not items:
            return
        
        items = []
        for anno in boxes:
            points = results.keypoints.xy.tolist()[0]
            points = Points2D(coords=points)
            points.to_int()

            anno.points = points
            items.append(anno)
        
        return items
        
    @staticmethod
    def from_segment(results: Results) -> list[Annotation]:
        if not results.masks:
            return
        
        boxes: list[Annotation] = ULParser.from_detect(results)
        if not boxes:
            return
        
        items = []
        for anno in boxes:
            points = results.masks.xy.tolist()[0]
            points = Points2D(coords=points)
            points.to_int()

            anno.points = points
            items.append(anno)

        return items
    