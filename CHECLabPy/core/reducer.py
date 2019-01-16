from inspect import getmembers
import numpy as np


class column:
    """
    Class to be used as a descriptor, identifying the columns to be added to
    the DL1 file.

    The `registry` attribute keeps track of all columns that have been defined.
    """
    registry = dict()

    def __init__(self, func=None):
        self.func = func
        self.__doc__ = func.__doc__

        if func.__name__ in self.registry:
            old_func = self.registry[func.__name__].__qualname__
            new_func = func.__qualname__
            if old_func != new_func:
                raise AttributeError("Duplicate WaveformReducer column "
                                     "name between {} and {}"
                                     .format(old_func, new_func))
        self.registry[func.__name__] = func

    def __get__(self, obj, obj_type=None):
        if obj is None:
            return self
        if self.func is None:
            raise AttributeError("unreadable attribute")
        if obj.waveforms is None:
            cn = obj.__class__.__name__
            raise ValueError("WaveformReducer {} not prepared".format(cn))
        return self.func(obj)


def _process_null(_):
    """
    Placeholder for an efficient replacement for when no columns of a
    `WaveformReducer` are activated.
    """
    return dict()


class WaveformReducerMeta(type):
    """
    Metaclass to define the `columns` attribute of a `WaveformReducer` before
    its initialisation.
    """
    def __new__(mcs, name, bases, dct):
        # Create the columns attribute
        columns = []
        for p, v in dct.items():
            # Check if member is a column
            if not isinstance(v, column):
                continue

            # Check if column is inherited from a parent
            try:
                for parent in bases:
                    parent_mem = dict(getmembers(parent))
                    if p in parent_mem and isinstance(parent_mem[p], column):
                        raise ValueError("Inherited")
            except ValueError:
                continue

            columns.append(p)
        dct['columns'] = columns

        return type.__new__(mcs, name, bases, dct)


class WaveformReducer(metaclass=WaveformReducerMeta):
    columns = None  # Created by metaclass

    def __init__(self, n_pixels, n_samples, **kwargs):
        """
        Base class for all WaveformReducers.

        Parameters
        ----------
        n_pixels : int
            Number of pixels in the data to be processed by the
            `WaveformReducer`
        n_samples : int
            Number of samples in the data to be processed by the
            `WaveformReducer`
        kwargs
            Columns can be deactivated by passing their "name"=False via the
            kwargs. Configuration to the `WaveformReducer` can also be
            passed via kwargs.
        """
        self.waveforms = None

        self.n_pixels = n_pixels
        self.n_samples = n_samples
        self.kwargs = kwargs

        self.active_columns = self.get_active_columns(**kwargs)

        if len(self.columns) == 0:
            self.process = _process_null

    @classmethod
    def get_active_columns(cls, **kwargs):
        """
        Parse the `kwargs` to check if the user has requested a column to be
        deactivated.

        Parameters
        ----------
        kwargs
            Columns can be deactivated by passing their "name"=False via the
            kwargs.

        Returns
        -------
        list
            List of the active columns
        """
        return [col for col in cls.columns if kwargs.get(col, True)]

    def _prepare(self, waveforms):
        """
        Method to prepare the `WaveformReducer` for processing the waveforms
        from a new event.

        By default this simply stores the waveforms in the `WaveformReducer`
        so they can be accessed by the columns. However, `WaveformReducer`
        subclasses are free to add additional functionality to this method.
        This is especially useful if multiple columns require the initial
        calculation or processessing, therefore repitition can be avoided if
        this is performed in the `_prepare` method.

        A copy is performed to ensure the `WaveformReducer` does not change
        the waveform in any way that could affect other `WaveformReducers`.

        Parameters
        ----------
        waveforms : ndarray
            The waveforms to be processed.
        """
        self.waveforms = np.copy(waveforms)

    def _get_dict(self):
        """
        Obtains the return values of all active columns, and stores them into
        a dict with a key corresponding to the name of the column.

        Returns
        -------
        dict
        """
        return {c: getattr(self, c) for c in self.active_columns}

    def _post(self):
        """
        Allows a process to be performed after the columns have been obtained.
        """
        pass

    def process(self, waveforms):
        """
        Process the waveforms of an event with this `WaveformReducer`.

        Parameters
        ----------
        waveforms : ndarray
            The waveforms to be processed.

        Returns
        -------
        d : dict
            Dictionary containing the return values of the columns, with a key
            corresponding to the name of the column.
        """
        self._prepare(waveforms)
        d = self._get_dict()
        self._post()
        return d
