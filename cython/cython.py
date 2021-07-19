""""    
Author: Bjarne Johannsen
Python Version: 3.8

Naive Cython Mandelbrot plotting.

Matrix Creation 1.34s, Calculation 36.31s @MacBook Air 2020 M1
"""

import cython
import time
import matplotlib.pyplot as plt
import h5py


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
    real = cython_naive_mandelbrot.allocate_matrix(REAL_MIN, REAL_MAX, SIZE, True)
    imag = cython_naive_mandelbrot.allocate_matrix(IMAG_MIN, IMAG_MAX, SIZE, False)
    stop = time.time()
    print('Matrix Allocation and Creation Time: ', str(round(stop - start, 2)))

    #calc Mandelbrot
    start = time.time()
    output_matrix = cython_naive_mandelbrot.calc_mandelbrot(real, imag, ITERATIONS, STABILITY_CRITERIA)
    stop = time.time()

    #plot figure with runtime and rescale bounds
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(output_matrix, cmap='viridis', extent=[REAL_MIN, REAL_MAX, IMAG_MIN, IMAG_MAX])
    ax.set_title('Naive Cython - Mandelbot Set: '+str(round(stop - start, 2))+' Seconds')
    plt.savefig('naiv_cython_'+str(SIZE)+'_'+str(ITERATIONS)+'_'+str(int(stop - start))+'s.pdf')
    plt.show()

    #save data
    file = h5py.File("naive_cython_mandelbrot.hdf5", "w")
    file.create_dataset('dataset', data=output_matrix)
    file.close()