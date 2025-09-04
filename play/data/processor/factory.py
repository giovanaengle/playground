from abc import ABC, abstractmethod
from dataclasses import dataclass

from play.common import Context
from play.data import Data

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
    def create(context: Context) -> Processor:
        process_ctx = context.sub('processor')
        process_ctx.logger.info(f'Creating processor')
        processes: list[Process] = []
        process_config = process_ctx.config.sub('process')
        for proc in process_config.dicts('processes'):
            if proc['process'] == 'crop':
                processes.append(CropProcess())
            elif proc['process'] == 'mask':
                processes.append(MaskProcess())
            elif proc['process'] == 'rename':
                processes.append(RenameProcess())
            elif proc['process'] == 'resize':
                processes.append(ResizeProcess(dimensions=proc['params']))

        processor: str = process_config.str('processor')
        if processor == 'linear':
            return LinearProcessor(processes=processes)
        else:
            raise Exception(f'unknown processor: {processor}')