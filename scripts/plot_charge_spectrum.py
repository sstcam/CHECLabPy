"""
A simple executable to demonstrate how to open the dl1 files and plot the
charge spectrum.
"""
import argparse
from argparse import ArgumentDefaultsHelpFormatter as Formatter
from matplotlib import pyplot as plt
from CHECLabPy.core.io import DL1Reader


def main():
    description = 'Plot the charge spectrum from a dl1 file'
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=Formatter)
    parser.add_argument('-f', '--file', dest='input_path', action='store',
                        required=True, help='path to the dl1 HDF5 run file')
    parser.add_argument('-p', '--pixel', dest='pixel', action='store',
                        type=int, default=0,
                        help='pixel to plot the spectrum of')
    args = parser.parse_args()

    input_path = args.input_path
    pixel = args.pixel

    with DL1Reader(input_path) as reader:
        pixel_arr, charge = reader.select_columns(['pixel', 'charge'])
        charge_pix = charge[pixel_arr == pixel]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    _, edges, _ = ax.hist(charge, bins=1000, histtype='step', normed=True,
                          label="All")
    ax.hist(charge_pix, bins=edges, histtype='step', normed=True,
            label="Pixel {}".format(pixel))
    ax.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    main()
