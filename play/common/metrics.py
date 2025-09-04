from play.utils.date import miliseconds

from .logger import Logger


class Metrics:
    def __init__(self, logger: Logger | None = None) -> None:
        self.logger = logger
        self.counters: dict[str, int] = {}
        self.starts: dict[str, int] = {}
        self.timers: dict[str, int] = {}

    def counter(self, name: str) -> int:
        return self.counters.get(name, 0)

    def dec(self, name: str, n: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) - n
        if self.logger:
            self.logger.debug(f'Decremented counter "{name}" to {self.counters[name]}')

    def inc(self, name: str, n: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + n
        if self.logger:
            self.logger.debug(f'Incremented counter "{name}" to {self.counters[name]}')

    def start(self, name: str) -> None:
        self.starts[name] = miliseconds()
        if self.logger:
            self.logger.debug(f'Started timer "{name}"')

    def stop(self, name: str) -> None:
        if name not in self.starts:
            if self.logger:
                self.logger.warning(f'Attempted to stop timer "{name}" which was never started')
            return

        elapsed = miliseconds() - self.starts[name]
        self.timers[name] = elapsed

        if self.logger:
            self.logger.info(f'Timer "{name}" completed in {elapsed} ms')

    def timer(self, name: str) -> int:
        return self.timers.get(name, 0)

    def summary(self) -> dict:
        return {
            'counters': self.counters.copy(),
            'starts': self.starts.copy(),
            'timers': self.timers.copy(),
        }
