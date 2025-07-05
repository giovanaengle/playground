from abc import ABC, abstractmethod
from typing import Text
from urllib.parse import ParseResult, urlparse

from data.image import Image


class Media(ABC):
    @abstractmethod
    def add(self) -> None:
        NotImplemented

    @abstractmethod
    def copy(self) -> None:
        NotImplemented
    
    @abstractmethod
    def download(self) -> None:
        NotImplemented

    @abstractmethod
    def load(self) -> None:
        NotImplemented
    
    @abstractmethod
    def save(self) -> None:
        NotImplemented

class MediaFactory:
    medias = {
        '.jpg': Image, 
        '.jpeg': Image,
        '.png': Image, 
        '.txt': Text,
    }

    @staticmethod
    def create(name: str) -> Media:
        if name in MediaFactory.medias:
            return MediaFactory.medias[name]
        else:
            raise Exception(f'Media type not accepted: {name}')