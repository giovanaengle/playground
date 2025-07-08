
from dataclasses import dataclass, field

from ..components import Data


@dataclass
class Job:
    current: list[Data]

    changes: list[Data] = field(default_factory=lambda: [])

    def process_changes(self) -> None:
        if len(self.changes) > 0:
            self.current = [data for data in self.changes]