from abc import ABC
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Label(ABC):
    coords: list[float | int] | None = None

    def denormalize(self, height: int | float, width: int | float) -> None:
        for idx, coord in enumerate(self.coords):
            if idx % 2 != 0:
                self.coords[idx] = int(coord * height)
            else:
                self.coords[idx] = int(coord * width) 

    def normalize(self, height: int | float, width: int | float) -> None:
        for idx, coord in self.coords:
            if idx % 2 != 0:
                self.coords[idx] = round(coord / height, 4)
            else:
                self.coords[idx] = round(coord / width, 4)
    
    def size(self) -> int:
        return len(self.coords)
    
    def to_array(self) -> np.ndarray:
        return np.array(self.coords)
    
    def to_float(self) -> None:
        float_bbox = [float(p) for p in self.coords] 
        self.coords = [round(f, 4) for f in float_bbox]
    
    def to_int(self) -> None:
        self.coords = [int(p) for p in self.coords] 
    
@dataclass
class Bbox(Label):
    orientation: float | None = None

    def area(self) -> list[int]:
        height = self.height()
        width = self.width()
        return round(height * width, 4)
                
    def denormalize(self, height: int | float, width: int | float) -> None:
        super().denormalize(height, width)

    def height(self) -> float:
        y1, y2 = self.coords[1], self.coords[3]
        height = y2 - y1
        return round(height, 4)
    
    def normalize(self, height: int | float, width: int | float) -> None:
        super().normalize(height, width)
    
    def to_array(self) -> np.ndarray:
        super().to_array()

    def to_float(self) -> None:
        super().to_float()
    
    def to_int(self) -> None:
        super().to_int()

    def to_xcyc(self) -> None:
        x1, y1, x2, y2 = self.coords
        h = y2 - y1
        w = x2 - x1
        xc = x1 + w / 2
        yc = y1 + h / 2
        coords = [xc, yc, w, h]
        self.coords = [round(p,4) for p in coords]

    def to_xyxy(self) -> None:
        xc, yc, w, h = self.coords
        x1 = xc - w/2
        x2 = xc + w/2
        y1 = yc - h/2
        y2 = yc + h/2
        self.coords = [x1, y1, x2, y2]

    def width(self) -> float:
        x1, x2 = self.coords[0], self.coords[2]
        width = x2 - x1
        return round(width, 4)

@dataclass
class Points2D(Label):

    def denormalize(self, height: int | float, width: int | float) -> None:
        super().denormalize(height, width)

    def height(self) -> float:
        ys = self.coords[1::2]
        max_y = max(ys)
        min_y = min(ys)
        height = max_y - min_y
        return round(height, 4)
    
    def normalize(self, height: int | float, width: int | float) -> None:
        super().normalize(height, width)
    
    def to_array(self) -> np.ndarray:
        super().to_array()

    def to_float(self) -> None:
        super().to_float()
    
    def to_int(self) -> None:
        super().to_int()
        
    def to_mask(self, image: np.ndarray) -> np.ndarray:
        filled = np.zeros_like(image)
        polygon = self.to_array(self.coords)
        mask = cv2.fillPoly(filled, pts=np.int32([polygon]), color=(255,255,255))
        return mask

    def width(self) -> float:
        xs = self.coords[1::2]
        max_x = max(xs)
        min_x = min(xs)
        width = max_x - min_x
        return round(width, 4)