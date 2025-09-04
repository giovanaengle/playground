
from pathlib import Path
from typing import Generator

import pandas as pd

from play.common import Context

from ..data import Data
from .ingest import Ingestor
from ..utils import Downloader


class CSVIngestor(Ingestor):
    '''
    CSV input format:
        - annotation: path to the txt file (local or URL)
        - image: path to the image file (local or URL)
        - text: path to the txt file (local or URL)
        - class: id (int) or name (str) of the class
    '''
    def __init__(self, context: Context, path: Path, downloader: Downloader) -> None:
        super().__init__(context=context, path=path)
        
        self.context.logger.info(f'Loading data from csv: {self.path}')
        self.df = pd.read_csv(self.path)
        self.downloader = downloader

        self.context.logger.info(f'Found total of {self.size()} inputs')

    def _parse_input(self, input: str) -> Path | None:
        if input.startswith('http'):
            parsed_input = self.downloader.download(input)
        else:
            parsed_input = Path(input)
        return parsed_input

    def load(self) -> Generator[Data, None, None]:
        columns = set(self.df.columns)  # faster lookup

        has_image = 'images' in columns
        has_text = 'texts' in columns
        has_annotation = 'annotations' in columns

        for row in self.df.itertuples(index=False):
            annotations, image, text = None, None, None
            name = ''

            if has_annotation and getattr(row, 'annotations'):
                if self.task == 'classify':
                    annotations = super()._load_class(row.annotations)
                else:
                    anno_path = self._parse_input(row.annotations)
                    annotations = super()._load_anno(anno_path)
                if not name:
                    name = annotations.name
            if has_image and getattr(row, 'images'):
                img_path = self._parse_input(row.images)
                image = super()._load_img(img_path)
                name = image.name
            if has_text and getattr(row, 'texts'):
                text_path = self._parse_input(row.texts)
                text = super()._load_text(text_path)
                if not name:
                    name = text.name

            yield Data(
                name=name,
                annotations=annotations,
                image=image,
                text=text,
            )

    def size(self) -> int:
        return self.df.shape[0]
