from argparse import ArgumentParser, Namespace
from traceback import print_exc

from common import Config
from core import Engine, EngineFactory


ENGINES: list[str] = [
    'dataset',
    'ingest',
    'export',
    'evaluate',
    'predict',
    'train',
    'validate'
]


def get_args() -> Namespace:
    parser: ArgumentParser = ArgumentParser(description='Pipeline')
    parser.add_argument('--engine', choices=ENGINES, help='The pipeline engine to run.', required=True, type=str)
    parser.add_argument('--project', help='The name of the project', required=True, type=str)
    return parser.parse_args()

def run(args: Namespace) -> None:
    config_path: str = f'.configs/{args.project}.yaml'
    config: Config = Config(path=config_path)
    engine: Engine = EngineFactory.create(config=config, engine=args.engine)
    engine.run()


if __name__ == '__main__':
    try:
        args: Namespace = get_args()
        run(args=args)
    except Exception:   
        print_exc()