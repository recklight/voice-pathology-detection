"""Seeding of the random number generators."""

from __future__ import annotations

import pytest

from voice_pathology_detection.seed import SeedRecord, seed_record, set_global_seed


def test_the_same_seed_gives_the_same_stream() -> None:
    first = set_global_seed(42)
    second = set_global_seed(42)
    assert first.integers(0, 1000, 5).tolist() == second.integers(0, 1000, 5).tolist()


def test_different_seeds_give_different_streams() -> None:
    a = set_global_seed(1).integers(0, 10**6, 5).tolist()
    b = set_global_seed(2).integers(0, 10**6, 5).tolist()
    assert a != b


def test_seeding_also_pins_the_standard_library() -> None:
    import random

    set_global_seed(7)
    first = [random.random() for _ in range(3)]
    set_global_seed(7)
    assert [random.random() for _ in range(3)] == first


def test_a_negative_seed_is_refused() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        set_global_seed(-1)


def test_the_record_says_what_was_seeded() -> None:
    record = seed_record(3)
    assert isinstance(record, SeedRecord)
    assert record.seed == 3
    assert record.numpy_seeded and record.random_seeded
    assert isinstance(record.torch_seeded, bool)
    assert record.to_dict()["seed"] == 3
