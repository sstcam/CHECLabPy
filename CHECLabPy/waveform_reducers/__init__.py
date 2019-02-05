"""
Module for containing all the `WaveformReducer` subclasses.
Each subclass should contain documentation describing the class.

To create a new `WaveformReducer` simply create a new file in this directory,
containing a class that inherits from `WaveformReducer`. The
`WaveformReducerChain` will automatically find any `WaveformReducer` inside
this directory, and add it as a link in the chain for the extract_dl1.py
executable.
"""

__all__ = []

import pkgutil
import inspect

for loader, name, is_pkg in pkgutil.walk_packages(__path__):
    module = loader.find_module(name).load_module(name)

    for key, value in inspect.getmembers(module):
        if name.startswith('__'):
            continue

        globals()[key] = value
        __all__.append(key)
