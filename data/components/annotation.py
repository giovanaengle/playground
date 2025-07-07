from dataclasses import dataclass, field

from .labels import Bbox, Points2D
from .media import Media


@dataclass
class Annotation:
    bbox: Bbox | None = None
    class_id: int | None = None
    class_name: str | None = None
    confidence: float | None = None
    points: Points2D | None = None

@dataclass
class Annotations(Media):
    items: list[Annotation] = field(default_factory=lambda: [])

    def add(self, anno: Annotation) -> None:
        self.items.append(anno)

    def clean(self) -> None:
        self.items.clear()

    def copy(self) -> 'Annotations':
        return Annotations(
            items=self.items.copy(),
            path=self.path,
        )
    
    def delete(self, anno: Annotation) -> None:
        self.items.remove(anno)

    def is_empty(self) -> bool:
        return len(self.items) <= 0 
    
    def load(self) -> None:
        if not self.parent or not self.is_empty:
            return

        path = self.parent.joinpath(f'{self.name}{self.suffix}')
        if not path.exists():
            raise Exception(f'Data not found in the designated path: {str(path)}')
        
        with open(path, 'r') as file:
            lines = file.readlines()

            if not lines:
                self.items.append(Annotation())
                return

            for line in lines:
                parts = line.replace('\n', '').split(' ')

                bbox = []
                points2D = []

                coords_length = len(parts[1:])
                if coords_length == 5:
                    bbox = Bbox(coords=[float(n) for n in parts[1:5]])
                    Bbox(orientation=parts[5])
                elif coords_length == 4:
                    bbox = Bbox(coords=parts[1:])
                    bbox.to_float()
                else:
                    points2D = Points2D([float(n) for n in parts[1:]])

                self.items.append(Annotation(
                    bbox=bbox,
                    class_id=int(parts[0]),
                    points=points2D,
                ))  

    def merge(self, config: dict[str, str]) -> None:
        items: list[Annotation] = self.copy()
        self.clean()

        classes = config['classes']
        merges = config['merges']
        for anno in items:
            if anno.class_id in merges:
                anno.class_id = merges[anno.class_id]
                anno.class_name = classes[anno.class_id]
                self.add(anno)
            else:
                self.add(anno)

    def save(self) -> None:
        path = self.parent.joinpath(self.name, self.suffix)
        with open(path, 'w', encoding='utf-8') as file:
            for anno in self.items:
                coords: list[str] = []
                if anno.points.size > 0:
                    coords = [f'{n}' for n in anno.points]
                elif anno.bbox.size > 0:
                    coords = [f'{n}' for n in anno.bbox]

                if anno.class_id is not None:
                    file.write(f'{anno.class_id} {' '.join(coords)}\n')