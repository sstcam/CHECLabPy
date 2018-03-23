from os.path import exists
from os import makedirs


def create_directory(directory):
    if directory:
        if not exists(directory):
            print("Creating directory: {}".format(directory))
            makedirs(directory)
