import os
from collections import defaultdict

import numpy as np
from aiorwlock import RWLock
from annoy import AnnoyIndex
from numpy import ndarray

from async_annoy import constants


class AnnoyIndexManager:
    """Manages read-write access to disk-based Annoy indices."""

    lock_storage: dict[str, RWLock] = defaultdict(RWLock)

    def __init__(
        self,
        unique_name: str,
        *,
        dimensions: int = constants.ASYNC_ANNOY_DIMENSIONS,
        metric: str = constants.ASYNC_ANNOY_METRIC,
    ):
        """Initialize a user-specific Annoy index."""
        self.name = unique_name
        self.dimensions = dimensions
        self.metric = metric
        self.ensure_directory_exists()
        self.lock = self.lock_storage[self.path]
        self.create_new_internal_instance()

    @property
    def path(self) -> str:
        """Get the path to the instance-specific Annoy index."""
        return os.path.join(
            constants.ASYNC_ANNOY_INDICES_DIRECTORY,
            f'{self.name}.ann',
        )

    def create_new_internal_instance(self):
        """Create a new Annoy index instance."""
        self.index = AnnoyIndex(self.dimensions, self.metric)

    def load_if_exists(self) -> bool:
        """Load the index memory-mapping from disk."""
        if os.path.exists(self.path):
            self.index.load(self.path)
            return True
        return False

    def ensure_directory_exists(self) -> None:
        """Ensure the directory for the index exists."""
        os.makedirs(constants.ASYNC_ANNOY_INDICES_DIRECTORY, exist_ok=True)


class AnnoyReader:
    """A context manager that allows read access to an Annoy index."""

    def __init__(
        self,
        unique_name: str,
        *,
        dimensions: int = constants.ASYNC_ANNOY_DIMENSIONS,
        metric: str = constants.ASYNC_ANNOY_METRIC,
    ):
        """Initialize the index manager."""
        self.manager = AnnoyIndexManager(
            unique_name,
            dimensions=dimensions,
            metric=metric,
        )
        if not self.manager.load_if_exists():
            raise ValueError(
                'Index %s does not exist.'
                % self.manager.path,
            )

    async def __aenter__(self):
        """Acquire a read lock."""
        await self.manager.lock.reader_lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Release a read lock."""
        self.manager.lock.reader_lock.release()

    async def get_neighbours_for(
        self,
        *,
        vector: ndarray,
        n: int = 3,  # noqa: WPS111
    ) -> list[int]:
        """Get the indices of the n nearest neighbors to a vector."""
        return self.manager.index.get_nns_by_vector(vector, n)

    async def get_vector_by(self, index: int) -> ndarray:
        """Get the vector at a given index."""
        return np.array(
            self.manager.index.get_item_vector(index),
            dtype=constants.DTYPE,
        )


class AnnoyWriter:
    """A context manager that allows write access to an Annoy index."""

    def __init__(
        self,
        unique_name: str,
        *,
        dimensions: int = constants.ASYNC_ANNOY_DIMENSIONS,
        metric: str = constants.ASYNC_ANNOY_METRIC,
    ):
        """Initialize the index manager."""
        self.manager = AnnoyIndexManager(
            unique_name,
            dimensions=dimensions,
            metric=metric,
        )

    async def __aenter__(self):
        """Acquire a write lock."""
        await self.manager.lock.writer_lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Build the index and release a write lock."""
        self.manager.index.build(10)
        self.manager.index.save(self.manager.path)
        self.manager.lock.writer_lock.release()

    async def add_item(self, index: int, vector: ndarray) -> None:
        """Add an item to the index."""
        self.manager.index.add_item(index, vector)
