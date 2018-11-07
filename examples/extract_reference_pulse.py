"""
Executable for extracting the reference pulse shape from the average of all
pixels and waveforms inside an R1 file, accounting for the time shift between
pixels.
"""


import argparse
from argparse import ArgumentDefaultsHelpFormatter as Formatter
import numpy as np
from CHECLabPy.core.io import ReaderR1
from CHECLabPy.plotting.waveform import WaveformPlotter
from CHECLabPy.utils.waveform import get_average_wf


def main():
    description = 'Extract the reference pulse shape from lab measurements'
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=Formatter)
    parser.add_argument('-f', '--file', dest='input_path', action='store',
                        required=True, help='path to the TIO r1 run file')
    parser.add_argument('-n', '--maxevents', dest='max_events', action='store',
                        help='Number of events to process', type=int)
    args = parser.parse_args()

    source = ReaderR1(args.input_path, args.max_events)
    t_shift = 60
    ref_pulse = get_average_wf(source, t_shift)
    ref_pulse /= np.trapz(ref_pulse)

    x_ref_pulse = np.arange(ref_pulse.size) * 1E-9

    p_wf = WaveformPlotter("Reference Pulse", "mV", "s")
    p_wf.add(x_ref_pulse, ref_pulse)
    p_wf.save("checs_reference_pulse.pdf")

    ref_save = np.column_stack((x_ref_pulse, ref_pulse))
    np.savetxt("checs_reference_pulse.txt", ref_save, fmt='%.5e')


if __name__ == '__main__':
    main()
