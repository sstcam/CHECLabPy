'''
Simple example showing how to loop over events from a single file, or multiple files
RW
'''

import numpy as np
from CHECLabPy.core.io import TIOReader

r0 = TIOReader('../CHECLabPy/data/chec_r1.tio')
r1 = TIOReader('../CHECLabPy/data/chec_r1.tio')

rlist = [r0, r1]
nev = rlist[0].n_events
print ("Number of events in file = %i" % nev)
for r in rlist:
    if not nev == r.n_events:
        print ("ERROR")

# Simple loop over events from 1 file
for iev in range(nev):
    wf = r0[iev]
    avwf = np.mean(wf)
    peak = np.max(avwf)
    tack = r0.current_tack
    if iev == 0: tack0 = tack
    print("Ev=%i #pix=%i #sam=%i dt=%d ms peak=%d" % (iev, len(wf), len(wf[0]), float(tack-tack0)/1e6, peak))

# Using itterators
for wf in r0:
    avwf = np.mean(wf)
    peak = np.max(avwf)
    tack = r0.current_tack
    if r0.index == 0: tack0 = tack
    print("Ev=%i #pix=%i #sam=%i dt=%d ms peak=%d" % (r0.index, len(wf), len(wf[0]), float(tack-tack0)/1e6, peak))

# Simple loop over events from 2 files
for iev in range(nev):
    print ("Ev=%i" % iev)
    for r in rlist:
        wf = r[iev]
        avwf = np.mean(wf)
        peak = np.max(avwf)
        tack = r.current_tack
        if iev == 0: tack0 = tack
        print("R%i #pix=%i #sam=%i dt=%d ms peak=%d" % (r.index, len(wf), len(wf[0]), float(tack-tack0)/1e6, peak))

# Alternative Method using itterators
for wfr0, wfr1 in zip(r0, r1):
    peakr0 = np.max(np.mean(wfr0))
    peakr1 = np.max(np.mean(wfr1))
    tack = r0.current_tack
    if r0.index == 0: tack0 = tack
    print ("Ev=%i" % r0.index)
    print ("R0 #pix=%i #sam=%i dt=%d ms peak=%d" % (len(wfr0), len(wfr0[0]), float(tack-tack0)/1e6, peakr0))
    print ("R1 #pix=%i #sam=%i dt=%d ms peak=%d" % (len(wfr1), len(wfr1[0]), float(tack-tack0)/1e6, peakr1))
