from dataclasses import dataclass

from ultralytics.utils.downloads import download
from yaml import dump

from data import Annotation, Data
from .dataset import Dataset, DatasetFormat


@dataclass
class ULDataset(Dataset):
    def _save_classify_data(self, data: Data, section: str) -> None:
        if not len(data.annotations.items):
            return

        anno: Annotation = data.annotations.items[0]
        if not anno.class_name:
            anno.class_name = self.classes[anno.class_id]

        data.image.name = data.name
        data.image.parent = self.path.joinpath(section, anno.class_name)
        data.image.save()

    def _save_classify(self) -> None:
        with open(self.path.joinpath('labels.txt'), 'w') as file:
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
        data.annotations.parent = self.path.joinpath(section, 'labels')
        data.image.name = data.name
        data.image.parent = self.path.joinpath(section, 'images')
        data.save()

    def _save_other(self) -> None:
        data = dict({
            'names': self.classes,
            'nc': len(self.classes),
            'test': f'{self.path.joinpath('test').absolute()}',
            'train': f'{self.path.joinpath('train').absolute()}',
            'val': f'{self.path.joinpath('valid').absolute()}',
        })
        with open(self.path.joinpath('data.yaml'), 'w') as file:
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
                self.path.joinpath(sub, cls).mkdir(exist_ok=True, parents=True)

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
            self.path.joinpath(t).mkdir(exist_ok=True, parents=True)
    
    def get(self, urls: list[str]) -> None:
        download(urls, dir=self.path)

    def prepare(self, data: list[Data]) -> None:
        super().prepare(data)

    def save(self) -> None:
        if self.format == DatasetFormat.CLASSIFY:
            self._save_classify()
        else:
            self._save_other()

    def setup(self) -> None:
        super().setup()

        if self.format == DatasetFormat.CLASSIFY:
            self._setup_classify()
        else:
            self._setup_other()
