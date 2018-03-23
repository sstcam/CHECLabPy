from inspect import isabstract
from CHECLabPy.core.base_reducer import WaveformReducer


class Factory:
    """
    Factory to provide a list of subclasses for a base class, and allow its
    selection at runtime.
    """
    subclasses = None  # Factory.child_subclasses(ParentClass)
    subclass_names = None  # [c.__name__ for c in subclasses]

    @staticmethod
    def child_subclasses(base):
        """
        Return all non-abstract subclasses of a base class.

        Parameters
        ----------
        base : class
            high level class object that is inherited by the
            desired subclasses

        Returns
        -------
        children : list
            list of non-abstract subclasses

        """
        family = base.__subclasses__() + [
            g for s in base.__subclasses__()
            for g in Factory.child_subclasses(s)
        ]
        children = [g for g in family if not isabstract(g)]

        return children

    @classmethod
    def produce(cls, product_name, **kwargs):
        factory = cls()
        subclass_dict = dict(zip(factory.subclass_names, factory.subclasses))

        try:
            product = subclass_dict[product_name]
        except KeyError:
            msg = ('No product found with name "{}" '
                   'for factory.'.format(product_name))
            raise KeyError(msg)

        return product(**kwargs)


class WaveformReducerFactory(Factory):
    import CHECLabPy.waveform_reducers
    subclasses = Factory.child_subclasses(WaveformReducer)
    subclass_names = [c.__name__ for c in subclasses]
