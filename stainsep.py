import numpy as np
from representative_region import select_representative_region
from BLTrans import BLTrans
from staincolor_hpcNMF import get_staincolor_hpcNMF

def stainsep(I, filename, magnification):
    rows, columns = I.shape[:2]

    I = select_representative_region(I, rows, columns, magnification)
    
    # transformada de Beer-Lamber
    V = BLTrans(I, filename)
    
    Wi = get_staincolor_hpcNMF()