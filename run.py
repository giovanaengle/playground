from argparse import ArgumentParser, Namespace
from pathlib import Path
from traceback import format_exc

from common import Context, LogLevel
from core import Engine, EngineFactory


DATA_ENGINES: list[str] = ['dataset', 'ingest']
MODEL_ENGINES: list[str] = ['export', 'evaluate', 'predict', 'train', 'validate']
ENGINES = sorted(DATA_ENGINES + MODEL_ENGINES)
LOGS: list[str] = ['debug', 'info', 'warning', 'error']


def get_args() -> Namespace:
    parser: ArgumentParser = ArgumentParser(description='AI Pipeline Runner')
    parser.add_argument('--engine', choices=ENGINES, help='The pipeline engine to run.', required=True, type=str)
    parser.add_argument('--project', help='The name of the project', required=True, type=str)
    parser.add_argument('--log-level', default='info', choices=LOGS)
    parser.add_argument('--log_to_console', default=True)
    parser.add_argument('--log_to_file', default=True)
    return parser.parse_args()

def main() -> int:
    args = get_args()
    engine_name = args.engine
    if engine_name in DATA_ENGINES:
        config_path: Path = Path(f'.configs/{engine_name}.yaml')
    else:
        config_path: Path = Path(f'.configs/model.yaml')
    try:
        context = Context(
            config_path=config_path,
            log_level=LogLevel[args.log_level.upper()],
            section=engine_name,
            log_to_console=args.log_to_console,
            log_to_file=args.log_to_file
        )
        context.logger.info(f'Running engine: {engine_name}')
        engine: Engine = EngineFactory.create(context, engine_name)
        engine.run()
    except Exception as ex:
        print(f'[FATAL] {type(ex).__name__}: {ex}')
        print(format_exc())
        return 1

    return 0


if __name__ == '__main__':
    exit(main())