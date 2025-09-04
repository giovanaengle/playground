import math

import cv2
import numpy as np

from play.data import Bbox


class ImageUtils:
    @staticmethod
    def align(array: np.ndarray, angle: float) -> np.ndarray:
        shape = array.shape[1::-1]
        center = tuple(np.ndarray(shape) / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(array, rotation_matrix, shape, flags=cv2.INTER_LINEAR)

    @staticmethod
    def compute_skew(array: np.ndarray) -> int:
        if array is not None and array.size > 0:
            h, w, _ = array.shape

            blurred_image = cv2.medianBlur(array, 3)
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

    @staticmethod
    def crop(array: np.ndarray, bbox: Bbox) -> np.ndarray:
        if not array.size == 0:
            x1, y1, x2, y2 = bbox.coords
            return array[y1 : y2, x1 : x2, :]

    @staticmethod
    def decode(array: np.ndarray) -> np.ndarray:
        if not array.size == 0:
            return cv2.imdecode(array, cv2.IMREAD_COLOR)

    @staticmethod
    def draw_circle(array: np.ndarray, center_coords: tuple[int,int], radius: int = 0, color: tuple[int,int,int] = (0,0,255), thickness: int = 100) -> np.ndarray:
        if not array.size == 0:
            return cv2.circle(array, center_coords, radius, color, thickness)

    @staticmethod
    def draw_rectangle(array: np.ndarray, bbox: Bbox, color: tuple[int,int,int] = (0,255,0), thickness: int = 5) -> np.ndarray:
        if not array.size == 0:
            x1, y1, x2, y2 = bbox.coords
            return cv2.rectangle(array, (x1, y1), (x2, y2), color, thickness)

    @staticmethod
    def encode(array: np.ndarray, suffix: str) -> np.ndarray:
        if not array.size == 0:
            _, encoded_image = cv2.imencode(suffix, array)
            return encoded_image

    @staticmethod
    def from_bytes(data: bytes) -> np.ndarray:
        return np.asarray(bytearray(data), dtype=np.uint8)

    @staticmethod
    def mask(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not array.size == 0:
            return cv2.bitwise_and(array, mask)

    @staticmethod
    def resize(array: np.ndarray, size: int | list[int], multiple: int | None = 32) -> np.ndarray:
        if not array.size == 0:
            proportional = False

            if isinstance(size, int):
                proportional = True
            elif len(size) == 1:
                size = size[0]
                proportional = True
            
            if not proportional:
                dsize = size
            else:
                shape = array.shape[:2] # h,w
                if shape[0] == shape[1]:
                    dsize = [size, size]
                else:
                    max_idx = shape.index(max(shape))
                    min_idx = shape.index(min(shape))

                    proportion_factor = size/shape[max_idx]
                    dsize = [0,0]
                    dsize[min_idx] = size
                    dsize[max_idx] = int(proportion_factor * shape[min_idx])

            if multiple:
                dsize = [max(math.ceil(x / multiple) * multiple, 0) for x in dsize]

            return cv2.resize(array, dsize=dsize, interpolation=cv2.INTER_AREA)

    @staticmethod
    def rotate(array: np.ndarray, angle: int) -> np.ndarray:
        if not array.size == 0:
            if angle == 90:
                rotate_code = cv2.ROTATE_90_CLOCKWISE
            elif angle == -90:
                rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
            elif angle == 180:
                rotate_code = cv2.ROTATE_180
            else:
                raise Exception(f'Rotated angle {angle} not acceptable')

            return cv2.rotate(array, rotate_code)

    @staticmethod
    def show(array: np.ndarray, legend: str = 'image') -> None:
        if not array.size == 0:
            cv2.imshow(legend, array)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @staticmethod
    def to_rgb(array: np.ndarray) -> np.ndarray:
        if not array.size == 0:
            return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    