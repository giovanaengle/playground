import argparse

from common import Config
from frameworks import FrameworkFactory
from models import ModelFactory


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    config = Config(path=args.config_path)

    model_config = config.sub('model')
    model_engine = ModelFactory.create(model_config)
    model_engine.set()

    framework_config = config.sub('framework')
    framework = FrameworkFactory.create(framework_config)
    print(framework)

