from ultralytics.utils.downloads import download
from yaml import dump

from common import Config
from data import Annotation, Data, TaskType
from .dataset import Dataset


class ULDataset(Dataset):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def _save_classify_data(self, data: Data, section: str) -> None:
        if not len(data.annotations.items):
            return

        anno: Annotation = data.annotations.items[0]
        if not anno.class_name:
            anno.class_name = self.classes[anno.class_id]

        data.image.name = data.name
        data.image.parent = self.output.joinpath(section, anno.class_name)
        data.image.save()

    def _save_classify(self) -> None:
        with open(self.output.joinpath('labels.txt'), 'w') as file:
            for class_name in self.classes:
                file.write(f'{class_name}\n')

        for data in self.test:
            self._save_classify_data(data, 'test')

        for data in self.train:
            self._save_classify_data(data, 'train')

        for data in self.valid:
            self._save_classify_data(data, 'val')

    def _save_other_data(self, data: Data, section: str) -> None:
        data.annotations.name = data.name
        data.annotations.parent = self.output.joinpath(section, 'labels')
        data.image.name = data.name
        data.image.parent = self.output.joinpath(section, 'images')
        data.save()

    def _save_other(self) -> None:
        data = dict({
            'names': self.classes,
            'nc': len(self.classes),
            'test': f'{self.output.joinpath('test').absolute()}',
            'train': f'{self.output.joinpath('train').absolute()}',
            'val': f'{self.output.joinpath('valid').absolute()}',
        })
        with open(self.output.joinpath('data.yaml'), 'w') as file:
            dump(data, file)

        for data in self.test:
            self._save_other_data(data, 'test')

        for data in self.train:
            self._save_other_data(data, 'train')

        for data in self.valid:
            self._save_other_data(data, 'valid')

    def _setup_classify(self) -> None:
        for sub in ['test', 'train', 'val']:
            for cls in self.classes:
                self.output.joinpath(sub, cls).mkdir(exist_ok=True, parents=True)

    def _setup_other(self) -> None:
        paths: list[str] = [
            'test/images',
            'test/labels',
            'train/images',
            'train/labels',
            'valid/images',
            'valid/labels',
        ]
        for t in paths:
            self.output.joinpath(t).mkdir(exist_ok=True, parents=True)
    
    def get(self, urls: list[str]) -> None:
        download(urls, dir=self.output)

    def prepare(self, data: list[Data]) -> None:
        super().prepare(data)

    def save(self) -> None:
        if self.format == TaskType.CLASSIFY:
            self._save_classify()
        else:
            self._save_other()

    def setup(self) -> None:
        super().setup()

        if self.format == TaskType.CLASSIFY:
            self._setup_classify()
        else:
            self._setup_other()
