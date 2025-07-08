from abc import ABC, abstractmethod
from uuid import uuid4

import numpy as np

from ..components import Bbox, Data, Points2D
from .job import Job


class Process(ABC):
    @abstractmethod
    def run(self, job: Job) -> None:
        NotImplemented

class CropProcess(Process):
    def run(self, job: Job) -> None:
        for data in job.current:
            for anno in data.annotations.items:
                if not anno.bbox:
                    continue

                bbox: Bbox = anno.bbox
                copy: Data = data.copy()
                height: int = copy.image.content.shape[1]
                width: int = copy.image.content.shape[0]

                bbox.denormalize(height, width)
                bbox.to_xyxy()
                copy.image.crop(bbox)
                
                job.changes.append(copy)

class MaskProcess(Process):
    def run(self, job: Job) -> None:
        for data in job.current:
            for anno in data.annotations.items:
                if not anno.points.size:
                    continue
                
                copy: Data = data.copy()
                height: int = copy.image.content.shape[1]
                points: Points2D = anno.points
                width: int = copy.image.content.shape[0]

                points.denormalize(height, width)
                mask: np.ndarray = points.to_mask(copy.image.content)
                copy.image.mask(mask)

                job.changes.append(copy)

class RenameProcess(Process):
    def run(self, job: Job) -> None:
        for data in job.current:
            copy: Data = data.copy()
            copy.name = uuid4().hex
            
            job.changes.append(copy)

class ResizeProcess(Process):
    dimensions: list[int]

    def __init__(self, dimensions: list[int]):
        super().__init__()

        self.dimensions = dimensions

    def run(self, job: Job) -> None:
        for data in job.current:
            copy: Data = data.copy()
            copy.image.resize(size=self.dimensions)

            job.changes.append(copy)