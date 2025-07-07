from dataclasses import dataclass

from .media import Media


@dataclass
class Text(Media):
    content: list[str] | None = None

    def copy(self) -> None:
        NotImplemented
    
    def download(self) -> None:
        NotImplemented
    
    def load(self) -> None:
        NotImplemented
    
    def save(self) -> None:
        NotImplemented