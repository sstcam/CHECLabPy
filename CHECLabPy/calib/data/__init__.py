from os.path import join, dirname


def get_calib_data(fn):
    return join(dirname(__file__), fn)
