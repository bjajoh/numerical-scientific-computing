""""    
Author: Bjarne Johannsen
Python Version: 3.8

Numpy vectorized, numba optimized and parallelized mandelbrot plotting.

Matrix Creation 0.05s, Calculation 4.9s @MacBook Air 2020 M1
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import math
import h5py
from numpy.core.fromnumeric import size
from numba import jit
import random
from multiprocessing import Pool
from functools import partial


def allocate_matrix(min_real, max_real, min_imag, max_imag, size):
    """"
    Allocate and prepare complex matrix using the numpy ogrid
    """
    y,x = np.ogrid[ min_imag:max_imag:size*1j, min_real:max_real:size*1j ]
    return x+y*1j

@jit(nopython=True)
def element_calc_mandelbrot(element, max_iter, threshold):
    """"
    Computation of the mandelbrot set for a single element.
    """
    z = complex(0, 0)  # start value set to zero
    for i in range(max_iter):
        z = z ** 2 + element  #calculation according to the algorithm
        if abs(z) > threshold:  #check stability
            return abs(z) / max_iter #set instability order
        elif i == max_iter - 1:  #stable
            return 1

def mulitcore_calc(matrix, threshold, max_iter, cores):
    """"
    Prepare Element wise calc function with args.
    Create a multiprocess pool with n cores.
    Map computation function to pool.
    """
    element_calc_mandelbrot_params = partial(element_calc_mandelbrot, max_iter=max_iter, threshold=threshold) 

    with Pool(processes=cores) as pool:
        output_matrix = pool.map(element_calc_mandelbrot_params, matrix,)
    
    return output_matrix


if __name__ == '__main__':
    REAL_MAX = 1
    REAL_MIN = -2
    IMAG_MIN = -1.5
    IMAG_MAX = 1.5
    SIZE = 5000  #Size of matrix --> size*size for matrix
    ITERATIONS = 1000  # Iterations to check the stability
    STABILITY_CRITERIA = 2  # Mandelbrot stability criteria 
    CORES = 8

    #alloc matrix
    start = time.time()
    complex_matrix = allocate_matrix(REAL_MIN, REAL_MAX, IMAG_MIN, IMAG_MAX, SIZE)
    complex_array = complex_matrix.flatten()
    stop = time.time()

    #calc Mandelbrot
    start = time.time()
    output_matrix = mulitcore_calc(complex_array, STABILITY_CRITERIA, ITERATIONS, CORES)
    stop = time.time()

    output_matrix = np.reshape(output_matrix, (SIZE, SIZE))

    #plot figure with runtime and rescale bounds
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(output_matrix, cmap='viridis', extent=[REAL_MIN, REAL_MAX, IMAG_MIN, IMAG_MAX])
    ax.set_title('Numpy Parallel Python - Mandelbot Set: '+str(round(stop - start, 2))+' Seconds with ' + str(CORES) + ' Cores')
    plt.savefig('numpy_numba_parallel_'+str(CORES)+'c_'+str(SIZE)+'_'+str(ITERATIONS)+'_'+str(int(stop - start))+'s.pdf')
    plt.show()

    #save data
    file = h5py.File("numpy_parallel_mandelbrot.hdf5", "w")
    file.create_dataset('dataset', data=SIZE)
    file.close()