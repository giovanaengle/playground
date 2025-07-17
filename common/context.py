from pathlib import Path
from typing import Optional

from .config import Config
from .logger import Logger, LogLevel
from .metrics import Metrics


class Context:
    def __init__(
        self,
        config_path: str | Path,
        log_level: LogLevel = LogLevel.INFO,
        section: str = 'main',
        log_to_console: bool = True,
        log_to_file: bool = True,
    ) -> None:
        config_path = str(config_path)

        # Set up logger first
        self.logger = Logger(
            level=log_level,
            section=section,
            log_to_console=log_to_console,
            log_to_file=log_to_file,
            config_path=config_path,
        )

        # Set up config
        self.config = Config(path=config_path)
        self.config.logger = self.logger  # Optional: attach logger manually

        # Set up metrics
        self.metrics = Metrics(logger=self.logger)

    def sub(self, section: str) -> 'Context':
        '''Creates a sub-context with a nested logger section (e.g., 'main:train')'''
        sub_logger = self.logger.clone(section)
        sub_ctx = Context.__new__(Context)
        sub_ctx.logger = sub_logger
        sub_ctx.config = self.config  # shared
        sub_ctx.metrics = Metrics(logger=sub_logger)
        return sub_ctx
