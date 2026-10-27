"""Worker pools that do not fight over the CPU.

NumPy's linear algebra library starts one thread per core in every process. With a pool of
N worker processes that is N times too many threads, and on a laptop it made dataset
generation several times *slower* than a single process (800 shapes: 48 s with 8 workers, 182 s
with 4, 30 s with one). Each worker should use one thread, so the pool is started with the
thread-count environment variables set to 1.

The workers are started with the ``spawn`` method on every platform. A forked worker would
inherit the parent's already-initialised library and ignore the variables. The cost is that
a script using a pool must guard its entry point with ``if __name__ == "__main__":``.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@contextmanager
def process_pool(workers: int) -> Iterator[ProcessPoolExecutor]:
    """A ``ProcessPoolExecutor`` whose workers each use a single math-library thread.

    The environment variables are restored on exit, so the calling process is unaffected.
    """
    saved = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
    os.environ.update({name: "1" for name in _THREAD_VARIABLES})
    try:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            yield pool
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
