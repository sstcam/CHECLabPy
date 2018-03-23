import pytest
from CHECLabPy.core.file_handling import ReaderR0, ReaderR1
from CHECLabPy.data import get_file


def test_readerr1():
    reader = ReaderR1(get_file("chec_r1.tio"))
    n_events = reader.n_events
    count = 0
    for _ in reader:
        count += 1

    assert count > 0
    assert count == n_events


def test_readerr1_getitem():
    reader = ReaderR1(get_file("chec_r1.tio"))
    event = reader[1]
    assert event.shape == (reader.n_pixels, reader.n_samples)
    assert reader.index == 1


def test_readerr1_single_module():
    reader = ReaderR1(get_file("targetmodule_r1.tio"))
    assert reader.n_pixels == 64
    assert reader[0].shape[0] == 64


def test_readerr0():
    reader = ReaderR0(get_file("targetmodule_r0.tio"))
    n_events = reader.n_events
    count = 0
    for _ in reader:
        count += 1

    assert count > 0
    assert count == n_events


def test_readerr0_with_r1():
    with pytest.raises(IOError):
        reader = ReaderR0(get_file("targetmodule_r1.tio"))


def test_readerr1_with_r0():
    with pytest.raises(IOError):
        reader = ReaderR1(get_file("targetmodule_r0.tio"))
