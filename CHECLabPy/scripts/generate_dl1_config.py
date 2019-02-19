"""
Create a fresh config file for the currently existing columns in the
`WaveformReducers`.

Also provides the documentation for each of the columns.
"""
from CHECLabPy.core.reducer import column, WaveformReducer
from CHECLabPy.core.chain import WaveformReducerChain
from CHECLabPy.core import child_subclasses
from CHECLabPy.data import get_file
import re
import argparse
import os
from argparse import ArgumentDefaultsHelpFormatter as Formatter


def main():
    description = ('Create a yaml config file for use with extract_dl1.py '
                   'to configure the WaveformReducers')
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=Formatter)
    parser.add_argument('-o', '--output', dest='output_path', action='store',
                        help='path to store the output HDF5 dl1 file')
    parser.add_argument('--bare', dest='bare', action='store_true',
                        help='create the file without docstrings')
    args = parser.parse_args()

    output_path = args.output_path or get_file("extractor_config.yml")
    bare = args.bare

    all_reducers = child_subclasses(WaveformReducer)
    default = WaveformReducerChain.default_columns

    with open(output_path, 'w') as f:
        for r in all_reducers:
            f.writelines("# {}\n".format(r.__name__))
            doc = r.__doc__
            if doc and not bare:
                doc = re.sub(r'(\n\s*)', '\n#  ', doc).strip()[:-1]
                f.writelines(doc)
            for c in r.columns:
                f.writelines("{}: {}\n".format(c, default.get(c, True)))
                doc = column.registry[c].__doc__
                if doc and not bare:
                    doc = re.sub(r'(\n\s*)', '\n#    ', doc).strip()[:-1]
                    f.writelines(doc)
            f.writelines("\n")

    print("Config file created: {}".format(output_path))


if __name__ == '__main__':
    main()
