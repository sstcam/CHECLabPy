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


def main():
    all_reducers = child_subclasses(WaveformReducer)
    default = WaveformReducerChain.default_columns

    path = get_file("extractor_config.yml")

    with open(path, 'w') as f:
        for r in all_reducers:
            f.writelines("# {}\n".format(r.__name__))
            doc = r.__doc__
            if r.__doc__:
                doc = re.sub(r'(\n\s*)', '\n#  ', doc).strip()[:-1]
                f.writelines(doc)
            for c in r.columns:
                f.writelines("{}: {}\n".format(c, default.get(c, True)))
                doc = column.registry[c].__doc__
                if doc:
                    doc = re.sub(r'(\n\s*)', '\n#    ', doc).strip()[:-1]
                    f.writelines(doc)
            f.writelines("\n")

    print("Config file created: {}".format(path))


if __name__ == '__main__':
    main()
