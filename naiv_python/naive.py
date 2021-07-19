
""""    
Author: Bjarne Johannsen
Python Version: 3.8

Simple and naiv mandelbrot implementation only with pure python.

Matrix Creation 4.04s, Calculation 61.78s @MacBook Air 2020 M1
"""

import time
import matplotlib.pyplot as plt
import h5py

def allocate_matrix(min, max, size, is_real):
    """allocate_matrix(min, max, size, is_real) -> matrix of size*size.
    Use same function for imag and read to enhance reusability."""
    matrix = []
    step_size = (abs(min) + max)/(size - 1)

    for i in range(size):
        row = []
        for j in range(size):
            if is_real: #check if real matrix or imag matrix should be created
                if j == 0:
                    row.append(min)
                else:
                    row.append(min + (step_size * j))
            else:   #create imaginary matrix
                if i == 0:
                    row.append(max)
                else:
                    row.append(max - (step_size * i))
        matrix.append(row)
    return matrix

def calc_mandelbrot(real_matrix, imaginary_matrix, max_iter, threshold):
    """"iterate over the matrix and check the mandelbrot stability.
    Naiv implemenation iteration over both matrices manually."""
    matrix = []

    #Iterate Rows
    for x in range(len(real_matrix)):
        real_row = real_matrix[x]
        imag_row = imaginary_matrix[x]
        temp_elements = []

        #iterate coloumn elements
        for y in range(len(real_matrix[0])):
            c = complex(real_row[y], imag_row[y])
            z = complex(0, 0)  # start value set to zero

            # iterate over element to check stability
            for i in range(max_iter):
                z = z ** 2 + c  # quadratic complex mapping
                # check unstability criteria reached
                if abs(z) > threshold:  
                    temp_elements.append(abs(z) / max_iter)
                    break
                # check element stable
                elif i == max_iter - 1:
                    temp_elements.append(1)

        # combine temp elements to matrix
        matrix.append(temp_elements)
    return matrix


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
    real_matrix = allocate_matrix(REAL_MIN, REAL_MAX, SIZE, True)
    imag_matrix = allocate_matrix(IMAG_MIN, IMAG_MAX, SIZE, False)
    stop = time.time()
    print('Matrix Allocation and Creation Time: ', str(round(stop - start, 2)))

    #calc Mandelbrot
    start = time.time()
    output_matrix = calc_mandelbrot(real_matrix, imag_matrix, ITERATIONS, STABILITY_CRITERIA)
    stop = time.time()

    #plot figure with runtime and rescale bounds
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(output_matrix, cmap='viridis', extent=[REAL_MIN, REAL_MAX, IMAG_MIN, IMAG_MAX])
    ax.set_title('Naive Python - Mandelbot Set: '+str(round(stop - start, 2))+' Seconds')
    plt.savefig('naiv_'+str(SIZE)+'_'+str(ITERATIONS)+'_'+str(int(stop - start))+'s.pdf')
    plt.show()

    #save data
    file = h5py.File("naive_mandelbrot.hdf5", "w")
    file.create_dataset('dataset', data=output_matrix)
    file.close()
