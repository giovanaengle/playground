from common import Config
from .hf import HfFramework
from .framework import Framework
from .ul import UlFramework
     
        
class FrameworkFactory:
    frameworks = {
        'hugging_face': HfFramework,
        'ultralytics': UlFramework,
    }
    
    @staticmethod
    def create(config: Config) -> Framework:
        name = config.str('name')
        if name in FrameworkFactory.frameworks:
            return FrameworkFactory.frameworks[name](config)
        else:
            raise Exception(f'Engine not found: {name}')
