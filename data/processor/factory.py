
from abc import ABC, abstractmethod
from dataclasses import dataclass

from common import Config
from ..components import Data
from .job import Job
from .process import (
    CropProcess,
    MaskProcess,
    Process,
    RenameProcess,
    ResizeProcess,
)


@dataclass
class Processor(ABC):
    processes: list[Process]

    @abstractmethod
    def process(self, data: Data) -> Job:
        NotImplemented

class LinearProcessor(Processor):
    def process(self, data: Data) -> Job:
        job: Job = Job(changes=[], current=[data])

        for proc in self.processes:
            proc.run(job)
            
            if len(job.changes) > 0:
                job.current = [data.copy() for data in job.changes]
                job.changes.clear()

        return job

class ProcessFactory:
    @staticmethod
    def create(config: Config) -> Processor:
        processes: list[Process] = []
        for proc in config.dicts('processes'):
            if proc['name'] == 'crop':
                processes.append(CropProcess())
            elif proc['name'] == 'mask':
                processes.append(MaskProcess())
            elif proc['name'] == 'rename':
                processes.append(RenameProcess())
            elif proc['name'] == 'resize':
                processes.append(ResizeProcess(dimensions=proc['params']))

        processor: str = config.str('processor')
        if processor == 'linear':
            return LinearProcessor(processes=processes)
        else:
            raise Exception(f'unknown processor: {processor}')