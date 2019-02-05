from CHECLabPy.core.io import ReaderR1
from CHECLabPy.data import get_file
from CHECLabPy.core.reducer import WaveformReducer, column
from CHECLabPy.core import child_subclasses
import CHECLabPy.waveform_reducers
import numpy as np
import pytest


def test_waveform_copy():
    """
    Test that the waveforms are not altered by any WaveformReducer
    """
    reader = ReaderR1(get_file("chec_r1.tio"))

    kwargs = dict(
        n_pixels=reader.n_pixels,
        n_samples=reader.n_samples,
        mapping=reader.mapping,
        reference_pulse_path=reader.reference_pulse_path,
    )

    waveforms = reader[0]
    test_waveforms = np.copy(waveforms)

    all_reducers = child_subclasses(WaveformReducer)
    for r in all_reducers:
        try:
            reducer = r(**kwargs)
            reducer.process(waveforms)

            if not (waveforms == test_waveforms).all():
                raise ValueError("WaveformReducer {} alters the waveforms!"
                                 .format(r.__name__))
        except ImportError:
            continue


class ExampleReducer(WaveformReducer):
    @column
    def test(self):
        return self.waveforms


class ExampleReducer2(ExampleReducer):
    @column
    def test2(self):
        return self.waveforms


def test_column():
    assert "test" in column.registry
    assert ["test"] == ExampleReducer.columns

    reducer = ExampleReducer(1, 1)
    assert ["test"] == reducer.columns
    assert ["test"] == reducer.active_columns

    reducer = ExampleReducer(1, 1, test=False)
    assert ["test"] == reducer.columns
    assert [] == reducer.active_columns


def test_process():
    reducer = ExampleReducer(1, 1)
    assert reducer.process(2) == dict(test=2)

    reducer = ExampleReducer(1, 1, test=False)
    assert reducer.process(2) == dict()


def test_duplication():
    with pytest.raises(AttributeError):
        class ExampleReducer3(WaveformReducer):
            @column
            def test(self):
                return self.waveforms


def test_inheritence():
    reducer = ExampleReducer2(1, 1)

    assert issubclass(ExampleReducer2, ExampleReducer)

    assert "test" not in ExampleReducer2.columns
    assert "test" not in reducer.columns
    assert "test" not in reducer.process(2)

    assert "test2" in ExampleReducer2.columns
    assert "test2" in reducer.columns
    assert "test2" in reducer.process(2)
