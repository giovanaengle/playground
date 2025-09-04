from pathlib import Path
from urllib.parse import ParseResult, urlparse
import urllib.request


class Downloader:
    def __init__(self, path: Path = Path('.projects/cache')):
        self.path = path
        self.path.mkdir(exist_ok=True, parents=True)

    def download(self, url: str) -> Path:
        name, suffix = self.parser_url(url)
        path: Path = self.path.joinpath(f'{name}.{suffix}')

        if not path.exists():
            urllib.request.urlretrieve(url, str(path))
        
        return path
    
    def parser_url(self, url: str) -> tuple[str, str]:
        url: ParseResult = urlparse(url=url)
        image: str = url.path.split('/')[-1]
        name, suffix = image.split('.')[:2]
        return (name, suffix)