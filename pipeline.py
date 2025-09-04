from argparse import ArgumentParser, Namespace
from pathlib import Path
from traceback import format_exc

from play.common import Context, LogLevel
from play.engine import Engine, EngineFactory


DATA_ENGINES: list[str] = ['dataset', 'ingest']
MODEL_ENGINES: list[str] = ['export', 'evaluate', 'predict', 'train', 'validate']
LOGS: list[str] = ['debug', 'error', 'info', 'warning']

ENGINES = sorted(DATA_ENGINES + MODEL_ENGINES)


def get_args() -> Namespace:
    parser: ArgumentParser = ArgumentParser(description='AI System')
    parser.add_argument('--config_path', help='The path to the yaml file configuration to run the system.', required=True, type=str)
    parser.add_argument('--engine', choices=ENGINES, help='The name of the engine to run.', required=True, type=str)
    
    parser.add_argument('--log-level', default='info', choices=LOGS)
    parser.add_argument('--log-to-file', action='store_true', help='Enable logging to a file.')
    
    return parser.parse_args()

def main() -> int:
    args = get_args()
    config_path = Path(args.config_path)
    engine_name = args.engine
    log_level = LogLevel.from_str(args.log_level)
    log_to_file = args.log_to_file

    try:
        context = Context(
            config_path=config_path,
            log_level=log_level,
            log_to_file=log_to_file,
            prefix=engine_name,
            section='main',
        )
            
        context.logger.info('Creating engine', engine=engine_name)
        engine: Engine = EngineFactory.create(context, engine_name)
        context.metrics.start(engine_name)
        engine.run()
        context.metrics.stop(engine_name)
        latency = context.metrics.timer(engine_name)
        context.logger.info(f'Engine run concluded', engine=engine_name, latency=latency)

    except Exception as ex:
        print(f'[FATAL] {type(ex).__name__}: {ex}')
        print(format_exc())
        return 1

    return 0


if __name__ == '__main__':
    exit(main())