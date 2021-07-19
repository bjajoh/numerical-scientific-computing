//Author: Bjarne Johannsen
//Python Version: 3.8
//OpenCL Mandelbrot plotting.
//Matrix Creation XXXs, Calculation XXXs @MacBook Air 2020 M1 <-- No support for OpenCL

#include <pyopencl-complex.h>       
  __kernel void mandelbrot(
    __global const float *real_matrix_device, 
    __global const float *imag_matrix_device, 
    __global       float *result_device,
             ushort const ITERATIONS,
             ushort const STABILITY_CRITERIA)

// Compute the Mandelbrot stability from imag and real matrix   
{     
  int id = get_global_id(0);
  float abs;
  cfloat_t z;  
  cfloat_t C; 
  C = cfloat_new(real_matrix_device[id], imag_matrix_device[id]); //Create complex number
  z = cfloat_new(0, 0); 
  bool flag = true;
  
  //Iterate to determince stability
  for(int i = 0; i < ITERATIONS; i++){
      Z = cfloat_mul(z,z); // z*z is the same as z^2
      Z = cfloat_add(z,c); // z = z+2
      abs = cfloat_abs(z);

      //Check stability and assign unstability or stability value
      if(abs > float(STABILITY_CRITERIA)){
          flag = false; 
          result_device[id] = abs / float(ITERATIONS); 
          break; 
      }
      if(i == ITERATIONS - 1){
          result_device[id] = 1;
      } 
  }
}
