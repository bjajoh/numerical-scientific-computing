""""    
Author: Bjarne Johannsen
Python Version: 3.8

Numpy vectorized and numba optimized mandelbrot plotting.

Matrix Creation 0.05s, Calculation 55.68s @MacBook Air 2020 M1
Matrix Creation 0.05s, Calculation 5.26s @MacBook Air 2020 M1 (numba)
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import math
import h5py
from numba import jit

def allocate_matrix(min_real, max_real, min_imag, max_imag, size):
    """"
    Allocate and prepare complex matrix using the numpy ogrid.
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


def calc_mandelbrot(matrix, threshold, max_iter):
    """"
    Calculate full mandelbrot matrix by vectorising the algo element wise.
    Numba can't parallelize vectorizations --> naiv loop 2x faster then vect.
    """
    vect_calc_mandelbrot = np.vectorize(element_calc_mandelbrot)
    return vect_calc_mandelbrot(matrix, max_iter, threshold)


if __name__ == '__main__':
    REAL_MAX = 1
    REAL_MIN = -2
    IMAG_MIN = -1.5
    IMAG_MAX = 1.5
    SIZE = 5000  #Size of matrix --> size*size for matrix
    ITERATIONS = 100  # Iterations to check the stability
    STABILITY_CRITERIA = 2  # Mandelbrot stability criteria 
    
    #alloc matrix
    start = time.time()
    complex_matrix = allocate_matrix(REAL_MIN, REAL_MAX, IMAG_MIN, IMAG_MAX, SIZE)
    stop = time.time()
    print('Matrix Allocation and Creation Time: ', str(round(stop - start, 2)))
  
    #calc Mandelbrot
    start = time.time()
    output_matrix = calc_mandelbrot(complex_matrix, STABILITY_CRITERIA, ITERATIONS)
    stop = time.time()

    #plot figure with runtime and rescale bounds
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(output_matrix, cmap='viridis', extent=[REAL_MIN, REAL_MAX, IMAG_MIN, IMAG_MAX])
    ax.set_title('Numpy Numba Python - Mandelbot Set: '+str(round(stop - start, 2))+' Seconds')
    plt.savefig('numpy_numba_'+str(SIZE)+'_'+str(ITERATIONS)+'_'+str(int(stop - start))+'s.pdf')
    plt.show()

    #save data
    file = h5py.File("numpy_mandelbrot.hdf5", "w")
    file.create_dataset('dataset', data=SIZE)
    file.close()
