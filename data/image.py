from dataclasses import dataclass, field
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import urllib.request

from .labels import Bbox
from .media import Media


img_formats: list[str] = ['.jpg', '.jpeg', '.png']


@dataclass
class Image(Media):
    name: str
    parent: Path
    suffix: str

    content: np.ndarray = field(default_factory=lambda: np.empty((0)))

    def align(self, angle: float) -> np.ndarray:
        if not self.is_empty():
            shape = self.content.shape[1::-1]
            center = tuple(np.ndarray(shape) / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(self.content, rotation_matrix, shape, flags=cv2.INTER_LINEAR)

    def mask(self, mask: np.ndarray) -> np.ndarray:
        if not self.is_empty():
            return cv2.bitwise_and(self.content, mask)

    def compute_skew(self) -> int:
        if not self.is_empty():
            h, w, _ = self.content.shape

            blurred_image = cv2.medianBlur(self.content, 3)
            edges = cv2.Canny(blurred_image,  threshold1 = 30,  threshold2 = 100, apertureSize = 3, L2gradient = True)
            lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30, minLineLength=w/4.0, maxLineGap=h/4.0)

            if lines is None:
                return 

            angle = 0
            cnt = 0
            for x1, y1, x2, y2 in lines[0]:
                ang = np.arctan2(y2 - y1, x2 - x1)
                if math.fabs(ang) <= 30: # excluding extreme rotations
                    angle += ang
                    cnt += 1

            if cnt == 0:
                return 0

            return int((angle/cnt) * 180 / math.pi)

    def copy(self) -> None:
        self.load()

        return Image(
            content=self.content.copy() if not self.is_empty() else np.empty((0)),
            name=self.name,
            parent=self.parent,
            suffix=self.suffix,
        )
    
    def crop(self, bbox: Bbox) -> np.ndarray:
        if not self.is_empty():
            x1, y1, x2, y2 = bbox.coords
            return self.content[y1 : y2, x1 : x2, :]

    def draw_circle(self, center_coords: tuple[int,int], radius: int = 0, color: tuple[int,int,int] = (0,0,255), thickness: int = 100) -> np.ndarray:
        if not self.is_empty():
            return cv2.circle(self.content, center_coords, radius, color, thickness)

    def draw_rectangle(self, bbox: Bbox, color: tuple[int,int,int] = (0,255,0), thickness: int = 5) -> np.ndarray:
        if not self.is_empty():
            x1, y1, x2, y2 = bbox
            return cv2.rectangle(self.content, (x1, y1), (x2, y2), color, thickness)

    def encode(self) -> np.ndarray:
        if not self.is_empty():    
            _, encoded_image = cv2.imencode(self.suffix, self.content)
            return encoded_image
        
    def is_empty(self) -> bool:
        return self.content.size <= 0
    
    def load(self) -> None:
        if self.is_empty():
            self.content = cv2.imread(self.path(), cv2.IMREAD_UNCHANGED)

    def path(self) -> Path:
        return self.parent.joinpath(f'{self.name}{self.suffix}')
    
    def resize(self, size: int | list[int], multiple: int | None = 32) -> np.ndarray:
        if not self.is_empty():
            proportional = False

            if isinstance(size, int):
                proportional = True
            elif len(size) == 1:
                size = size[0]
                proportional = True

            if proportional:
                shape = self.content.shape[:2] # h,w
                if shape[0] == shape[1]:
                    size = [size, size]
                else:
                    max_idx = shape.index(max(shape))
                    min_idx = shape.index(min(shape))

                    proportion_factor = size/shape[max_idx]
                    size = [0,0]
                    size[min_idx] = size
                    size[max_idx] = int(proportion_factor * shape[min_idx])

            if multiple:
                size = [max(math.ceil(x / multiple) * multiple, 0) for x in size]

            return cv2.resize(self.content, dsize=size, interpolation=cv2.INTER_AREA)

    def rotate(self, angle: int) -> np.ndarray:
        if not self.is_empty():
            if angle == 90:
                rotate_code = cv2.ROTATE_90_CLOCKWISE
            elif angle == -90:
                rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
            elif angle == 180:
                rotate_code = cv2.ROTATE_180
            else:
                raise Exception(f'Rotated angle {angle} not acceptable')

            return cv2.rotate(self.content, rotate_code)

    def save(self) -> None:
        if not self.is_empty():
            cv2.imwrite(self.path(), self.content)

    def show(self, legend: str = 'image') -> None:
        if not self.is_empty():
            cv2.imshow(legend, self.content)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def to_rgb(self) -> np.ndarray:
        if not self.is_empty():
            return cv2.cvtColor(self.content, cv2.COLOR_BGR2RGB)