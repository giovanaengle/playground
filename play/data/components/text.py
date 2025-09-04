from dataclasses import dataclass

from .component import Component


@dataclass
class Text(Component):
    content: list[str] | None = None

    def copy(self) -> None:
        NotImplemented
    
    def download(self) -> None:
        NotImplemented
    
    def load(self) -> None:
        NotImplemented
    
    def save(self) -> None:
        NotImplemented