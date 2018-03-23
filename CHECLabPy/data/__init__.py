from os.path import join, dirname


def get_file(fn):
    return join(dirname(__file__), fn)
