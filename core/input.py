
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import pandas as pd

from common import Config
from data import Annotation, Annotations, Data, Image, Text
from .download import Downloader


@dataclass
class Input(ABC):
    path: Path

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

class CSVInput(Input):
    df: pd.DataFrame
    downloader: Downloader

    def __init__(self, path: Path, downloader: Downloader) -> None:
        super().__init__(path=path)
        self.df = pd.read_csv(self.path)
        self.downloader = downloader

    def _parse_input(self, path: str) -> Path:
        if path.startswith('http'):
            path = self.downloader.download(path)
        else:
            path = Path(path)
        return path

    def load(self) -> Generator[Data, None, None]:
        columns = self.df.columns.tolist()
        for _, row in self.df.iterrows():
            annotations, image, text = None, None, None

            if 'image' in columns:
                img_path = self._parse_input(row['image'])
                image = super()._load_img(img_path)
                name = image.name

            if 'text' in columns:
                text_path = self._parse_input(row['text'])
                text = super()._load_text(text_path)
                name = text.name

            if 'annotation' in columns:
                anno_path = self._parse_input(row['annotation'])
                annotations = super()._load_anno(anno_path)
                name = annotations.name

            data: Data = Data(
                annotations=annotations,
                image=image,
                name=name,
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
        
        self.files.extend(self.path.glob('images/*.jpg'))
        self.files.extend(self.path.glob('images/*.jpeg'))
        self.files.extend(self.path.glob('images/*.png'))
        
        if not len(self.files):
            self.files.extend(self.path.glob('texts/*.txt'))
        if not len(self.files):
            self.files.extend(self.path.glob('annotations/*.txt'))
        if not len(self.files):
            raise Exception(f'Data not found in the path structure: {path}. Data must be in on of the folders: images, labels, or texts')

    def load(self) -> Generator[Data, None, None]:
        path: Path
        for path in self.files:
            annotations, image, text = None, None, None

            data: Data
            if path.parent.name == 'images':
                image = self._load_img(path=path)
                path = str(path).replace(path.suffix, '.txt')
                path = Path(path.replace('images', 'annotations'))
                if path.exists():
                    annotations = self._load_anno(path=path)
                path = Path(str(path).replace('annotations', 'texts'))
                if path.exists():
                    text = self._load_text(path=path)
                name = image.name
            elif path.parent.name == 'texts':
                text = self._load_text(path=path)
                path = Path(str(path).replace('texts', 'annotations'))
                if path.exists():
                    annotations = self._load_anno(path=path)
                name = text.name

            elif path.parent.name == 'annotations':
                annotations = self._load_anno(path=path)
                name = annotations.name

            data: Data = Data(
                annotations=annotations,
                image=image,
                name=name,
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
                downloader=Downloader(),
            )
        elif path.is_dir():
            return DirInput(
                path=path,
            )
        else:
            raise Exception(f'Data path format not accepted: {str(path)} \n', 'It must be csv or directory type')