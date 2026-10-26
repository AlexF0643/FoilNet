import os

import pytest

from foilnet.parallel import process_pool

NAMES = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]


def test_workers_start_with_single_threaded_math_libraries(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    with process_pool(2) as pool:
        assert list(pool.map(os.getenv, NAMES)) == ["1", "1", "1"]


def test_the_callers_environment_is_restored(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    with process_pool(1) as pool:
        list(pool.map(abs, [1]))
        assert os.environ["OMP_NUM_THREADS"] == "1"  # set while the pool is in use
    assert os.environ["OMP_NUM_THREADS"] == "7"
    assert "OPENBLAS_NUM_THREADS" not in os.environ


def test_the_environment_is_restored_when_the_body_raises(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    with pytest.raises(RuntimeError), process_pool(1):
        raise RuntimeError("boom")
    assert os.environ["OMP_NUM_THREADS"] == "7"


def test_results_come_back_in_order():
    with process_pool(2) as pool:
        assert list(pool.map(abs, [-3, -1, -2, 0, 5])) == [3, 1, 2, 0, 5]
