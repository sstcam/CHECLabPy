from CHECLabPy.core.reducer import WaveformReducer, column
from CHECLabPy.core.chain import WaveformReducerChain


class ExampleReducer(WaveformReducer):
    @column
    def test(self):
        return self.waveforms


class ExampleReducer2(WaveformReducer):
    @column
    def test2(self):
        return self.waveforms


def test_chain():
    # Disable existing columns
    config = {}
    for c in column.registry.keys():
        if c is not "test" and c is not "test2":
            config[c] = False

    chain = WaveformReducerChain(1, 1, **config)
    assert len(chain.chain) == 2
    assert chain.process(2) == dict(test=2, test2=2)

    chain = WaveformReducerChain(1, 1, test=False, **config)
    assert len(chain.chain) == 1
    assert chain.process(2) == dict(test2=2)

    chain = WaveformReducerChain(1, 1, test=False, test2=False, **config)
    assert len(chain.chain) == 0
    assert chain.process(2) == dict()
