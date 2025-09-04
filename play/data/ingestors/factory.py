from pathlib import Path

from play.common import Context

from .csv import CSVIngestor
from .dir import DirIngestor
from ..utils import Downloader
from .ingest import Ingestor


class IngestorFactory:
    @staticmethod
    def create(context: Context) -> Ingestor:
        input_ctx = context.sub('input')
        input = context.config.str('input')
        parent = context.config.path('parent')
        project = context.config.str('project')
        path: Path = parent.joinpath(project, input)
        if not path.exists():
            message = f'Input path does not exist: {str(path)}'
            input_ctx.logger.error(message)
            raise ValueError(message)
        
        if path.suffix == '.csv':
            return CSVIngestor(
                context=input_ctx,
                downloader=Downloader(),
                path=path,
            )
        elif path.is_dir():
            return DirIngestor(
                context=input_ctx,
                path=path,
            )
        else:
            message = f'Path must be csv or directory, got instead: {str(path)}'
            input_ctx.logger.error(message)
            raise ValueError(message)