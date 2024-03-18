# async-annoy README

## Overview
`async-annoy` simplifies working with Annoy indices in async applications like web servers. It's built to let you read from or write to Annoy indices without worrying about the complexities of concurrent access. You get easy, thread-safe operations, whether you're querying nearest neighbors or building the index, all in the background.

## Features
- **Automated Index Management**: Handles all the tricky parts of managing Annoy indices on disk.
- **Concurrency Made Simple**: Supports multiple readers at once or a single writer by managing access locks behind the scenes.

## Installation
```bash
pip install async-annoy
```
Make sure Annoy and numpy are installed as they're needed to run the library.

## How to Use
### Reading from the Index
Ideal for fetching nearest neighbors in response to web requests.
```python
from async_annoy import AnnoyReader

async with AnnoyReader("my_index") as reader:
    neighbours = await reader.get_neighbours_for(vector=my_vector, n=5)
    print(neighbours)
```
### Writing to the Index
The writer will wait for all readers to finish and start the index from scratch.
The Annoy library doesn't support index updates, only rebuilds.
```python
from async_annoy import AnnoyWriter

async with AnnoyWriter("my_index") as writer:
    await writer.add_item(index=1, vector=my_vector)
    # The index is automatically built and saved when done.
```
## Configuration
The initial setup of `async-annoy` relies on environment variables to configure the Annoy index parameters, ensuring a seamless start. While these defaults offer convenience, you may need to customize settings to fit your application's specific requirements. It's critical to maintain parameter consistency across all readers and writers interacting with the same index.

### Default Parameters
`async-annoy` uses the following environment variables for initial configuration:
- `ASYNC_ANNOY_DIMENSIONS`: The dimensionality of the vectors stored in the index.
- `ASYNC_ANNOY_METRIC`: The distance metric used for comparing vectors in the index (e.g., "angular", "euclidean").

### Overriding Default Parameters
Although `async-annoy` configures itself with environment variables, you can override these defaults directly in your code. When creating a new `AnnoyReader` or `AnnoyWriter`, simply pass the desired dimensions and metric as arguments:

```python
from async_annoy import AnnoyReader, AnnoyWriter

# Override default parameters when initializing
reader = AnnoyReader("my_index", dimensions=128, metric="euclidean")
writer = AnnoyWriter("my_index", dimensions=128, metric="euclidean")
```
### Important Note
Ensure consistency in the parameters used by readers and writers. Mismatches in dimensions or metrics can lead to incorrect results or runtime errors. Every instance that accesses the same index must use the same dimensions and metric settings

### Development Status
Currently, async-annoy is in alpha. It's ready for testing, and we're eager for your feedback to make it better. This is an initial version, so we're still working on adding more features and smoothing out the experience.
