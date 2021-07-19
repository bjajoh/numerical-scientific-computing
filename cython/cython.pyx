""""    
Author: Bjarne Johannsen
Python Version: 3.8

Naive Cython Mandelbrot plotting.

Matrix Creation 1.34s, Calculation 36.31s @MacBook Air 2020 M1
"""
import matplotlib.pyplot as plt


cpdef allocate_matrix(min, max, size, is_real):
    """allocate_matrix(min, max, size, is_real) -> matrix of size*size"""
    cdef double step_size = (abs(min) + max)/(size - 1)
    cdef int i
    cdef int j
    matrix = []

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

    
cpdef calc_mandelbrot(real_matrix, imaginary_matrix, int max_iter, int threshold):
    """"iterate over the matrix and check the mandelbrot stability"""
    matrix = []
    cdef int x
    cdef int y
    cdef int i
    cdef double complex c
    cdef double complex Z

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