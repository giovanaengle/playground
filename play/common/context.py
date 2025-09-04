from pathlib import Path

from .config import Config
from .logger import Logger, LogLevel
from .metrics import Metrics


class Context:
    def __init__(
        self,
        config_path: str | Path,
        log_level: LogLevel = LogLevel.INFO,
        log_to_file: bool = False,
        prefix: str | None = None,
        section: str = 'main') -> None:

        self.config = Config(path=config_path)
        if log_to_file:
            project = self.config.str('project')
            file_path = self.config.path('parent').joinpath(project, 'logs')
        else:
            file_path = None
        self.logger = Logger(
            path=file_path,
            log_level=log_level,
            prefix=prefix,
            section=section,
        )
        self.metrics = Metrics(logger=self.logger)

    def sub(self, section: str) -> 'Context':
        '''Creates a sub-context with a nested logger section (e.g., 'main:train')'''
        sub_logger = self.logger.clone(section)
        sub_ctx = Context.__new__(Context)
        sub_ctx.logger = sub_logger
        sub_ctx.config = self.config  # shared
        sub_ctx.metrics = Metrics(logger=sub_logger)
        return sub_ctx
