""""    
Author: Bjarne Johannsen
Python Version: 3.8

Dask parallelized mandelbrot plotting.

Even for small matrix sized is the runtime extremly high, 
due to the large overhead and too simple problem.
Furthermore numba not supported.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from dask.distributed import Client, wait
import dask.delayed as delay
import dask.array as da
import webbrowser
import h5py


def allocate_matrix(min_real, max_real, min_imag, max_imag, size):
    """"
    Allocate and prepare complex matrix using the numpy ogrid
    """
    y,x = np.ogrid[ min_imag:max_imag:size*1j, min_real:max_real:size*1j ]
    return x+y*1j


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


def calc_array_dask(array, threshold, iterations, nbr_workers):
    """"
    Computation of the mandelbrot set for a single element.
    """
    client = Client(n_workers=nbr_workers)
    result = client.map(element_calc_mandelbrot, array, [iterations]*len(array), [threshold]*len(array))

    map_array = client.gather(result)  # gather the results from the clients

    client.close()

    return map_array


if __name__ == '__main__':
    WORKERS = 4  # Number of workers used for processing
    SIZE = 50  # Square matrix dimension
    ITERATIONS = 100  # Number of iterations for mandelbrot computation
    STABILITY_CRITERIA = 2
    REAL_MIN = -2
    REAL_MAX = 1
    IMAG_MIN = -1.5
    IMAG_MAX = 1.5
    
    #alloc matrix
    start = time.time()
    complex_matrix = allocate_matrix(REAL_MIN, REAL_MAX, IMAG_MIN, IMAG_MAX, SIZE)
    stop = time.time()
    print('Matrix Allocation and Creation Time: ', str(round(stop - start, 2)))

    #calc Mandelbrot
    start = time.time()
    # Convert to flattend dask array, compute and restore shape
    dask_array = da.from_array(complex_matrix.flatten(), chunks=(1000))
    map_array = calc_array_dask(dask_array, STABILITY_CRITERIA, ITERATIONS, WORKERS)
    output_matrix = np.reshape(map_array, (SIZE, SIZE))
    stop = time.time()

    #plot figure with runtime and rescale bounds
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(output_matrix, cmap='viridis', extent=[REAL_MIN, REAL_MAX, IMAG_MIN, IMAG_MAX])
    ax.set_title('Dask Python - Mandelbot Set: '+str(round(stop - start, 2))+' Seconds with ' + str(WORKERS) + ' Workers')
    plt.savefig('numpy_dask_'+str(WORKERS)+'w_'+str(SIZE)+'_'+str(ITERATIONS)+'_'+str(int(stop - start))+'s.pdf')
    plt.show()

    #save data
    file = h5py.File("dask_mandelbrot.hdf5", "w")
    file.create_dataset('dataset', data=SIZE)
    file.close()
