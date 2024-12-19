import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import prange
from libc.math cimport log10, sqrt, round
cnp.import_array()

#-------------------------------------------------------
cdef inline int imin(int a, int b) nogil:
    return a if a < b else b

#-------------------------------------------------------
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def dens_weight_scale( cnp.ndarray[cnp.float32_t, ndim=3] density, int nmesh, int num_threads=96 ):
    
    cdef int i, j, k
    cdef int   rp4   = 1
    cdef int   rm4   = 10
    cdef float const = ( rp4*log10(1e4) - rm4*log10(1e-4) )/ (rm4-rp4)
    cdef float r0    = rp4*( log10(1e4)+const )
    cdef float[:,:,:] density_view = density
    cdef cnp.ndarray result= np.empty((nmesh,nmesh,nmesh), dtype=np.int32) 
    cdef int[:,:,:] result_view = result

    # Calculate scale array for the entire density field
    for i in prange(nmesh, nogil=True, num_threads=num_threads):
        for j in range(nmesh):
            for k in range(nmesh):
                if density_view[i,j,k] <= 1.e-4:
                    result_view[i,j,k] = rm4 
                else:
                    result_view[i,j,k] = <int> round( r0/(log10(density_view[i,j,k]) + const) )
    return result_view

#-------------------------------------------------------
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def dens_weight_smooth( cnp.ndarray[cnp.float32_t, ndim=3] field, cnp.ndarray[cnp.float32_t, ndim=3] density, int nmesh, int num_threads=96 ):
    
    cdef int i, j, k, i_min, i_max, j_min, j_max, k_min, k_max, scale
    cdef int i_, j_, k_, ii, jj, kk, di, dj, dk, ncount
    cdef cnp.ndarray field_R   = np.zeros( (nmesh,nmesh,nmesh), dtype=np.float32 )
    #cdef cnp.ndarray scale_arr = np.empty( (nmesh,nmesh,nmesh), dtype=np.int32 ) 
    cdef float[:,:,:] field_view   = field 
    cdef float[:,:,:] density_view = density    
    cdef float[:,:,:] field_R_view = field_R
    cdef int[:,:,:]   scale_arr_view 

    # Calculate the scale for the entire density array using the dens_weight_scale function
    scale_arr_view  = dens_weight_scale(density, nmesh, num_threads)
  
    # Iterate over each position
    for i in prange(nmesh, nogil=True, num_threads=num_threads):
        for j in range(nmesh):
            for k in range(nmesh):
                scale  = scale_arr_view[i,j,k]
                ncount = 0
                for i_ in range(i-scale, i+scale+1):
                    for j_ in range(j-scale, j+scale+1):
                        for k_ in range(k-scale, k+scale+1):
                            ii, jj, kk = i_ % nmesh, j_ % nmesh, k_ % nmesh
                            di, dj, dk = ii-i, jj-j, kk-k
                            if (di<0):
                                di = -di
                            else:
                                di = di
                            if (dj<0):
                                dj = -dj
                            else:
                                dj = dj
                            if (dk<0):
                                dk = -dk
                            else:
                                dk = dk
                            
                            di         = imin(di, nmesh-di)
                            dj         = imin(dj, nmesh-dj)
                            dk         = imin(dk, nmesh-dk)
                            if <int> round( sqrt(di**2+dj**2+dk**2) )<=scale:
                                field_R_view[i,j,k] += field_view[ii,jj,kk]
                                ncount         += 1
                field_R_view[i,j,k] /= ncount
    
    return field_R_view
