from datetime import datetime, timezone
from enum import Enum
import json
import logging
from pathlib import Path
import sys
from typing import Optional

from rich.logging import RichHandler

from play.api.context import request_id_ctx


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            'time': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'prefix': getattr(record, 'prefix', ''),
            'section': getattr(record, 'section', ''),
            'message': record.getMessage(),
        }
        if request_id_ctx.get():
            log_obj['request_id'] = request_id_ctx.get()
        if hasattr(record, 'extra_data'):
            log_obj.update(record.extra_data)
        return json.dumps(log_obj)

class LogLevel(int, Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR

    def from_str(level: str) -> 'LogLevel':
        try:
            return LogLevel[level.upper()]
        except KeyError:
            raise ValueError(f'Unknown log level: {level}')

class Logger:
    base_logger: Optional[logging.Logger] = None

    def __init__(
        self,
        path: Path | None = None,
        log_level: LogLevel = LogLevel.INFO,
        prefix: str | None = None,
        section: str = 'root',
    ) -> None:
        
        if Logger.base_logger is not None:
            pass 
        
        self.path = path
        self.log_level = log_level
        self.prefix = prefix
        self.section = section or 'root'
        
        Logger.base_logger = logging.getLogger(f'{prefix or "root"}:{section}')
        Logger.base_logger.setLevel(log_level.value)
        Logger.base_logger.propagate = False  # Avoid duplicate logs in global root

        # StreamHandler for logs in console
        rich_handler = RichHandler(rich_tracebacks=True, markup=True)
        rich_handler.setLevel(log_level.value)
        Logger.base_logger.addHandler(rich_handler)

        if self.path:
            formatter = JsonFormatter()
            if self.path.is_file():
                if not self.path.suffix == '.log':
                    raise ValueError(f'Format to logs path not known')
            else:
                self.path.mkdir(exist_ok=True, parents=True)
                ts: str = datetime.now().strftime('%Y%m%d-%H%M%S')
                self.path = self.path.joinpath(f'{ts}.log')

            file_handler = logging.FileHandler(self.path)
            file_handler.setFormatter(formatter)
            Logger.base_logger.addHandler(file_handler)

        self.logger = Logger.base_logger

    @property
    def handlers(self):
        return self.logger.handlers
    
    @property
    def level(self):
        return self.logger.level

    def _log(self, log_level: LogLevel, message: str, **kwargs) -> None:
        exc_info = kwargs.pop('exc_info', None)  # extract if provided
        self.logger.log(
            log_level,
            message,
            extra={
                'prefix': self.prefix,
                'section': self.section,
                'extra_data': kwargs,
            },
            exc_info=exc_info,
        )

    def clone(self, section: str) -> 'Logger':
        return Logger(path=self.path, log_level=self.log_level, prefix=self.prefix, section=section)

    def clone_handler(self, handler: logging.Handler) -> logging.Handler:
        if isinstance(handler, logging.StreamHandler):
            new_handler = logging.StreamHandler(sys.stdout)
        elif isinstance(handler, logging.FileHandler):
            new_handler = logging.FileHandler(handler.baseFilename)
        else:
            raise ValueError(f'Cloning not supported for handler {type(handler)}')
        new_handler.setLevel(handler.level)
        new_handler.setFormatter(handler.formatter)
        return new_handler

    def debug(self, message: str, **kwargs) -> None:
        self._log(LogLevel.DEBUG, message, **kwargs)

    def error(self, message: str | Exception, **kwargs) -> None:
        self._log(LogLevel.ERROR, message, **kwargs)

    def exception(self, message: str, **kwargs) -> None:
        self._log(LogLevel.ERROR, message, exc_info=True, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        self._log(LogLevel.INFO, message, **kwargs)

    def sub(self, prefix: str) -> 'Logger':
        if self.prefix:
            prefix = f'{self.prefix}:{prefix}'
        return Logger(path=self.path, log_level=self.log_level, prefix=prefix, section=self.section)

    def warning(self, message: str, **kwargs) -> None:
        self._log(LogLevel.WARNING, message, **kwargs)
