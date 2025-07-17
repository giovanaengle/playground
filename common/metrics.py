from typing import Optional

from .logger import Logger
from utils.date import miliseconds


class Metrics:
    def __init__(self, logger: Optional[Logger] = None) -> None:
        self.logger = logger
        self.counters: dict[str, int] = {}
        self.timers: dict[str, int] = {}

    def inc(self, name: str, n: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + n
        if self.logger:
            self.logger.debug(f"Incremented counter '{name}' to {self.counters[name]}")

    def dec(self, name: str, n: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) - n
        if self.logger:
            self.logger.debug(f"Decremented counter '{name}' to {self.counters[name]}")

    def counter(self, name: str) -> int:
        return self.counters.get(name, 0)

    def start(self, name: str) -> None:
        self.timers[name] = miliseconds()
        if self.logger:
            self.logger.debug(f"Started timer '{name}'")

    def stop(self, name: str) -> None:
        if name not in self.timers:
            if self.logger:
                self.logger.warning(f"Attempted to stop timer '{name}' which was never started")
            return

        elapsed = miliseconds() - self.timers[name]
        self.timers[name] = elapsed

        if self.logger:
            self.logger.info(f"Timer '{name}' completed in {elapsed} ms")

    def timer(self, name: str) -> int:
        return self.timers.get(name, 0)

    def summary(self) -> dict:
        return {
            "counters": self.counters.copy(),
            "timers": self.timers.copy(),
        }
