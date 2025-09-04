from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from play.common import Context
from play.data import Annotation, Annotations, Data, Image, Text


@dataclass
class Ingestor(ABC):
    def __init__(self, context: Context, path: Path) -> None:
        self.context = context
        self.path = path
        self.task = context.config.str('task')

    def _load_anno(self, path: Path) -> Annotations:
        items = []
        if path.suffix:
            parent=path.parent
        else:
            labels = str(path)
            if ',' in labels:
                content = labels.split(';')
            else:
                content = [labels]

            for label in content:
                anno = Annotation(
                    class_name=label
                )
                items.append(anno)
            
            parent = None
        
        annotations: Annotations = Annotations(
            items=items,
            name=str(path.stem),
            parent=parent,
            suffix=path.suffix
        )

        return annotations

    def _load_class(self, cls: int | str) -> Annotations:
        classes = self.context.config.strs('classes')
        try:
            if isinstance(cls, int):
                class_id = cls
                class_name = classes[class_id]
            else:
                class_name = cls
                class_id = classes.index(class_name)
        except Exception as e:
            self.context.logger.exception(f'Class not found: {cls}', exc_info=e)
        
        anno = Annotation(
            class_id=int(class_id),
            class_name=str(class_name)
        )
        annotations: Annotations = Annotations(
            items=[anno],
            name='',
            parent='',
            suffix='',
        )
        return annotations

    def _load_img(self, path: Path) -> Image:
        image: Image = Image(
            name=str(path.stem),
            parent=path.parent,
            suffix=path.suffix,
        )

        return image

    def _load_text(self, path: Path) -> Text:
        if path.suffix:
            content = None
            parent = path.parent
        else:
            texts = str(path)
            if ',' in texts:
                content = texts.split(';')
            else:
                content = [texts]
            parent = None

        text: Text = Text(
            content=content,
            name=str(path.stem),
            parent=parent,
            suffix=path.suffix
        )

        return text
    
    @abstractmethod
    def load(self) -> Generator[Data, None, None]:
        NotImplemented

    @abstractmethod
    def size(self) -> int:
        NotImplemented