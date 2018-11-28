from CHECLabPy.core import child_subclasses
from CHECLabPy.core.reducer import WaveformReducer
from CHECLabPy.core.spectrum_fitter import SpectrumFitter


class Factory:
    """
    Factory to provide a list of subclasses for a base class, and allow its
    selection at runtime.
    """
    subclasses = None  # Factory.child_subclasses(ParentClass)
    subclass_names = None  # [c.__name__ for c in subclasses]

    @classmethod
    def produce(cls, product_name, *args, **kwargs):
        print("Obtaining {} from {}".format(product_name, cls.__name__))
        factory = cls()
        subclass_dict = dict(zip(factory.subclass_names, factory.subclasses))

        try:
            product = subclass_dict[product_name]
        except KeyError:
            msg = ('No product found with name "{}" '
                   'for factory.'.format(product_name))
            raise KeyError(msg)

        return product(*args, **kwargs)


class WaveformReducerFactory(Factory):
    import CHECLabPy.waveform_reducers
    subclasses = child_subclasses(WaveformReducer)
    subclass_names = [c.__name__ for c in subclasses]


class SpectrumFitterFactory(Factory):
    import CHECLabPy.spectrum_fitters
    subclasses = child_subclasses(SpectrumFitter)
    subclass_names = [c.__name__ for c in subclasses]
