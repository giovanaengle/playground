from abc import ABC, abstractmethod


class Media(ABC):
    @abstractmethod
    def copy(self) -> None:
        NotImplemented
    
    @abstractmethod
    def load(self) -> None:
        NotImplemented
    
    @abstractmethod
    def save(self) -> None:
        NotImplemented