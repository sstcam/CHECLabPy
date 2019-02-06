'''
Convert CHEC dl1 hd5 file to ROOT TTREE

RW d2018-03-14 - Initial version
RW d2018-04-04 - Added CHECLabPy Reader

-- Requires root_numpy :( 
-- On the MPIK lab PC this can be enabled with "source activate pyroot"
-- On linux the following seems to work to create the correct env (but not on Mac)

conda create --name=pyroot root=6 python=3 pandas numpy root-numpy
conda install root-numpy

e.g. see https://nlesc.gitbooks.io/cern-root-conda-recipes/content/installing_root-numpy_and_rootpy_via_conda.html

TODO:
* Finish monitor item conversion once dl1 format is fixed
'''

import os
import argparse
import time
import numpy as np
import pandas as pd
from ROOT import TFile, TTree, gROOT, AddressOf
from root_numpy import array2tree
gROOT.ProcessLine("struct StringHolder{Char_t fString[256];};")
from ROOT import StringHolder
from CHECLabPy.core.io import DL1Reader

def Write2TTree(data, tName):
    print ("--> Writing to TTree: %s", tName)
    t0 = time.time()
    print ("    * Converting to structrured array")
    data_arr = data.to_records(index=False)
    print ("       dt=%i s" % (time.time()-t0))    
    print("    * Converting to TTree")
    t0 = time.time()
    ttree = array2tree(data_arr, name=tName)
    print ("       dt=%i s" % (time.time()-t0))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert hd5 to ROOT TTree')
    parser.add_argument('-i', '--in', dest='fname_hd5', action='store',
                        required=True, help='DL1 (Pandas) hd2f filename with path (required)')
    parser.add_argument('-o', '--out', dest='fname_root', action='store', required=False,
                        help='ROOT output filename with path (default: input.root)', default=None)
    args = parser.parse_args()

    fname_hd5 = args.fname_hd5
    fname_root = args.fname_root

    if not os.path.exists(fname_hd5):
        print ("ERROR: -i (--in) file %s does not exist" % fname_hd5)
        exit()

    if fname_root == None:
        fname_root = os.path.splitext(fname_hd5)[0] + '.root'
    
    print ("--> Converting %s to %s" % (fname_hd5, fname_root))

    print ("--> Opening HD5 file")
    dl1 = DL1Reader(fname_hd5)
    
    print ("--> Loading Data")
    data = dl1.load_entire_table()
    meta = dl1.metadata
    mapping = dl1.mapping
    mon = None
    if hasattr(dl1,'monitor'):
        dl1.monitor.load_entire_table()    
        print ("--> Found the following monitor data:")
        for key in mon:
            print ("\t%s = %s (type=%s)" % (key, str(mon[key]), type(mon[key])))
    
    print ("--> Found the following meta data:")
    for key in meta:
        print ("\t%s = %s (type=%s)" % (key, str(meta[key]), type(meta[key])))
        if type(meta[key]) == pd._libs.tslib.Timestamp:
            meta[key] = meta[key].asm8.view(np.int64)
            print("\tTimestamp found, converting to uint64: \t%s = %s (type=%s)" % (key, str(meta[key]), type(meta[key])))

    print ("--> Found the following data columns:")
    for col in data.columns:
        print ("\t%s (%s)" % (col, data[col].dtype))
        if col == 't_cpu':
            data[col] = data[col].view(np.int64)
            print("\tTimestamp found, converting to uint64: \t%s (%s)" % (col, data[col].dtype))

    print ("--> Opening root file")
    f = TFile(fname_root, 'recreate')

    print("--> Writing meta data to TTree: tMeta")
    tmeta = TTree('tMeta','tMeta')
    meta_numeric = meta.copy()
    meta_strings = []
    for key in meta:
        val = meta[key]
        if val == None: val = "None"
        if type(val) == str:
            meta_strings.append(StringHolder())
            tmeta.Branch(key, AddressOf(meta_strings[-1], 'fString'), key + '/C')
            meta_strings[-1].fString = val
            del meta_numeric[key]
    tmeta.Fill()
    meta_arr = pd.DataFrame(meta_numeric, index=[0]).to_records(index=False)
    array2tree(meta_arr, tree = tmeta)

    Write2TTree(data, 'tData')
    Write2TTree(mapping, 'tMapping')
    if mon: Write2TTree(mon, 'tMonitor')
    
    print ("--> Closing files")
    f.Write()
    f.Close()
