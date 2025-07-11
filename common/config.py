from pathlib import Path
from typing import Any

from yaml import safe_load


class Config:
    items: dict[str, Any]

    def __init__(self, items: dict[str, Any] = {}, path: str | Path | None = None) -> None:
        self.items = items

        if path:
            with open(path, 'r') as file:
                self.items = safe_load(file)

    def bool(self, key: str) -> bool:
        if not key in self.items:
            return False
        return self.items[key]

    def dict(self, key: str) -> dict:
        return self.items[key]

    def dicts(self, key: str) -> list[dict]:
        value: list[dict] = self.items[key]
        return value

    def floats(self, key: str) -> list[float]:
        return self.items[key]

    def ints(self, key: str) -> list[int]:
        return self.items[key]

    def path(self, key: str) -> Path:
        return Path(self.items[key])

    def str(self, key: str) -> str:
        return self.items[key]

    def strs(self, key: str) -> list[str]:
        return self.items[key]

    def sub(self, prefix: str) -> 'Config':
        return Config(items=self.items[prefix])