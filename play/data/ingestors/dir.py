from collections import defaultdict
from pathlib import Path
from typing import Generator

from play.common import Context
from play.data import Data

from .ingest import Ingestor


class DirIngestor(Ingestor):
    '''
    Directory structure: 
        Classify:
            - <dir>/images/<class_name1>/image_files.<ext>
            - <dir>/images/<class_name2>/image_files.<ext>
        Multimodal:
            - <dir>/annotations/anno_files.txt
            - <dir>/images/image_files.<ext>
            - <dir>/texts/text_files.txt
    '''
    ANNO_EXTS = (".txt",)
    IMAGE_EXTS = (".jpg", ".jpeg", ".png")
    TEXT_EXTS = (".txt",)

    FOLDERS = ('annotations', 'images', 'texts')

    def __init__(self, context: Context, path: Path) -> None:
        super().__init__(context=context, path=path)
        self.files: list[Path] = []
        
        self.context.logger.info(f'Loading data from directory: {self.path}')
        if self.task == 'classify':
            self._collect_classify_files()
        else:
            self._collect_multimodal_files()

        try:
            folder_names = [f.name for f in self.path.iterdir() if f.is_dir() and f.name in self.FOLDERS]
        except Exception:
            msg = f'None of the required folders were found in the directory: {self.path}'
            self.context.logger.error(msg)
            raise FileNotFoundError(msg)
        
        self.context.logger.info(f'Found files in the folders: {folder_names}')
        self.context.logger.info(f'Ingesting {self.size()} samples into the pipeline')

    def _collect_classify_files(self) -> None:
        img_dir = self.path.joinpath('images')
        if not img_dir.exists():
            msg = f'Image directory does not exist: {img_dir}. Required for classification task.'
            self.context.logger.error(msg)
            raise NotADirectoryError(msg)
        
        for ext in self.IMAGE_EXTS:
            self.files.extend(img_dir.glob(f'**/*{ext}'))
        self.context.logger.debug(f'Collected {len(self.files)} images at: {img_dir}')
        
        if not self.files:
            message = f'No images found at: {img_dir}. Check the format of your directory or the task.'
            self.context.logger.error(message)
            raise FileNotFoundError(message)

    def _collect_multimodal_files(self) -> None:
        img_dir = self.path.joinpath('images')
        txt_dir = self.path.joinpath('texts')
        anno_dir = self.path.joinpath('annotations')

        if anno_dir.exists():
            for ext in self.ANNO_EXTS:
                self.files.extend(anno_dir.glob(f'**/*{ext}'))
            self.context.logger.debug(f'Collected annotations from: {anno_dir}')
        
        if img_dir.exists():
            for ext in self.IMAGE_EXTS:
                self.files.extend(img_dir.glob(f'**/*{ext}'))
            self.context.logger.debug(f'Collected images from: {img_dir}')
        
        if txt_dir.exists():
            for ext in self.TEXT_EXTS:
                self.files.extend(txt_dir.glob(f'**/*{ext}'))
            self.context.logger.debug(f'Collected texts from: {txt_dir}')

        if not self.files:
            message = f'No images, texts, or annotations found at: {self.path}. Check the directory format.'
            self.context.logger.error(message)
            raise ValueError(message)

    def load(self) -> Generator[Data, None, None]:
        grouped: dict[str, dict[str, Path]] = defaultdict(dict)
        for filepath in self.files:
            stem = filepath.stem
            parent = filepath.parent.name

            if parent == 'annotations':
                grouped[stem]['annotations'] = filepath
            elif parent == 'images':
                grouped[stem]['image'] = filepath
            elif parent == 'texts':
                grouped[stem]['text'] = filepath

        for stem, parts in grouped.items():
            data = Data(name=stem)

            if 'annotations' in parts:
                if self.task == 'classify':
                    cls = parts['annotations'].parent.name
                    data.annotations = self._load_class(cls)
                else:
                    data.annotations = self._load_anno(parts['annotations'])
            if 'image' in parts:
                data.image = self._load_img(parts['image'])
            if 'text' in parts:
                data.text = self._load_text(parts['text'])

            yield data

    def size(self, all: bool = False) -> int:
        '''Return number of unique samples (by stem) or total files if flag "all" is set.'''
        if all:
            return len(self.files)
        return len({f.stem for f in self.files})
