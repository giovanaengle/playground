
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import pandas as pd

from common import Config
from data import Annotation, Annotations, Data, Image, Text


@dataclass
class Input(ABC):
    path: Path

    name: str | None = None

    def _load_anno(self, path: Path) -> Annotations:
        items = []
        if path.suffix:
            parent=path.parent
            if not self.name:
                self.name = path.stem
        else:
            labels = str(path)
            if ',' in labels:
                content = labels.split(';')
                for label in content:
                    anno = Annotation(
                        class_name=label
                    )
                    items.append(anno)
            else:
                items = [labels]
            parent = None

        annotations: Annotations = Annotations(
            items=items,
            name=self.name,
            parent=parent,
            suffix=path.suffix
        )

        return annotations

    def _load_img(self, path: Path) -> Image:
        if not self.name:
            self.name = str(path.stem)

        image: Image = Image(
            name=self.name,
            parent=path.parent,
            suffix=path.suffix,
        )

        return image

    def _load_text(self, path: Path) -> Text:
        if path.suffix:
            content = None
            parent = path.parent
            if not self.name:
                self.name = path.stem
        else:
            texts = str(path)
            if ',' in texts:
                content = texts.split(';')
            else:
                content = [texts]
            parent = None

        text: Text = Text(
            content=content,
            name=self.name,
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

class CSVInput(Input):
    df: pd.DataFrame

    def __init__(self, path: Path) -> None:
        super().__init__(path=path)
        self.df = pd.read_csv(self.path)

    def load(self) -> Generator[Data, None, None]:
        for _, row in self.df.iterrows():
            annotations, image, text = None, None, None

            if 'image' in row:
                img_path = str(row['image'])
                image = super()._load_img(img_path)
            if 'text' in row:
                text = str(row['text'])
                text = super()._load_text(text)
            if 'annotations' in row:
                annotations = str(row['annotations'])
                annotations = super()._load_anno(annotations)
            else:
                raise Exception(f'No data found: missing columns annotation, image, or text in the csv file {self.path}')

            data: Data = Data(
                annotations=annotations,
                image=image,
                name=self.name,
                text=text,
            )
            yield data

    def size(self) -> int:
        return self.df.shape[0]

class DirInput(Input):
    files: list[Path]

    def __init__(self, path: Path) -> None:
        super().__init__(path=path)
        self.files = []
        self.path = path

        self.files.extend(self.path.glob('**/*.jpg'))
        self.files.extend(self.path.glob('**/*.jpeg'))
        self.files.extend(self.path.glob('**/*.png'))

        if not self.files:
            self.files.append(self.path.glob('texts/*.txt'))
        if not self.files:
            self.files.append(self.path.glob('labels/*.txt'))
        if not self.files:
            raise Exception(f'Data not found in the path structure: {path} \n', 'Data must be in on of the folders: images, labels, or texts')
            
    def load(self) -> Generator[Data, None, None]:
        path: Path
        for path in self.files:
            annotations, image, text = None, None, None

            data: Data
            if path.parent.name == 'images':
                image = self._load_img(path=path)
                path = str(path).replace(path.suffix, '.txt')
                path = Path(path.replace('images', 'labels'))
                if path.exists():
                    annotations = self._load_anno(path=path)
                path = Path(str(path).replace('labels', 'texts'))
                if path.exists():
                    text = self._load_text(path=path)

            elif path.parent.name == 'texts':
                text = self._load_text(path=path)
                path = Path(str(path).replace('texts', 'labels'))
                if path.exists():
                    annotations = self._load_anno(path=path)

            elif path.parent.name == 'labels':
                annotations = self._load_anno(path=path)
            
            data: Data = Data(
                annotations=annotations,
                image=image,
                name=self.name,
                text=text,
            )
            yield data

    def size(self) -> int:
        return len(self.files)

class InputFactory:
    @staticmethod
    def create(config: Config) -> Input:
        path: Path = config.path('input')
        if path.suffix == '.csv':
            return CSVInput(
                path=path,
            )
        elif path.is_dir():
            return DirInput(
                path=path,
            )
        else:
            raise Exception(f'Data path format not accepted: {str(path)} \n', 'It must be csv or directory type')