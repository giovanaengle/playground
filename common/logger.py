# common/logger.py

import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class LogLevel(int, Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR


class Logger:
    def __init__(
        self,
        level: LogLevel = LogLevel.INFO,
        section: Optional[str] = None,
        log_to_console: bool = True,
        log_to_file: bool = True,
        config_path: Optional[str] = None,
    ) -> None:
        self.section = section or 'root'
        self.logger = logging.getLogger(self.section)
        self.logger.setLevel(level.value)
        self.logger.propagate = False  # Avoid duplicate logs in global root

        formatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)s] :%(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Clear old handlers if reused (e.g. in notebooks or tests)
        self.logger.handlers.clear()

        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        if log_to_file:
            log_dir = Path('.pipeline/logs')
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            log_file = log_dir / f'{timestamp}.log'

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Optional: log the config path (helpful for debugging runs)
        if config_path:
            self.info(f'Loaded configuration from: {config_path}')

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str | Exception) -> None:
        self.logger.error(str(message))

    def clone(self, section: str) -> 'Logger':
        '''Create a nested logger with an extended section name.'''
        full_section = f'{self.section}:{section}'
        return Logger(
            level=LogLevel(self.logger.level),
            section=full_section,
            log_to_console=True,
            log_to_file=True,
        )
