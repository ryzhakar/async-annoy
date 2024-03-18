from os import getenv

from dotenv import load_dotenv
from numpy import float32

load_dotenv()

_COMMON_EMBEDDING_DIMENSIONS: int = 384
DTYPE = float32
ASYNC_ANNOY_INDICES_DIRECTORY: str = getenv(
    'ASYNC_ANNOY_INDICES_DIRECTORY',
    'async_annoy_indices',
)
ASYNC_ANNOY_DIMENSIONS: int = int(
    getenv(
        'ASYNC_ANNOY_DIMENSIONS',
        _COMMON_EMBEDDING_DIMENSIONS,
    ),
)
ASYNC_ANNOY_METRIC: str = getenv(
    'ASYNC_ANNOY_METRIC',
    'angular',
)
