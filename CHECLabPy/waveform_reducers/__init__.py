"""
Module for containing all the waveform reduction approachess.
Each method should contain documentation describing the class.

To create a new waveform reducer simply create a new file in this directory,
containing a class that inherits from `WaveformReducer`. The
`WaveformReducerFactory` will automatically find any `WaveformReducer` inside
this directory, and add it as an option to the extract_dl1.py executable.

In the new waveform reducer you may override the default
`core.base_reducer.WaveformReducer` method. The storage of parameters
extracted from the reducer is very flexible, therefore users can define new
parameters to return from the reducer in the dict, and they will be stored in
the dl1 file.
"""

__all__ = []

import pkgutil
import inspect

for loader, name, is_pkg in pkgutil.walk_packages(__path__):
    module = loader.find_module(name).load_module(name)

    for name, value in inspect.getmembers(module):
        if name.startswith('__'):
            continue

        globals()[name] = value
        __all__.append(name)
