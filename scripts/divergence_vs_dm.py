import bigfile
import numpy as np
from dens_weight_smooth_para.dens_weight_smooth import dens_weight_smooth

# Parameters used to specify the path
simulation = "/TNG"      
run        = "/TNG50-1"  
redshift   = "/z0"
matter     = "gas"    

# Load the fields
dens_path = '/...'+simulation+run+redshift+'/dens_fields/dm/dens_field_pcs.bigfile'
div_path = '/...'+simulation+run+redshift+'/div_fields/'+matter+'/div_field_pcs.bigfile'
with bigfile.File(dens_path) as bf:
    shape       = bf['Field'].attrs['ndarray.shape']
    dens_rfield = bf['Field'][:].reshape(shape)

with bigfile.File(div_path) as bf:
    shape      = bf['Field'].attrs['ndarray.shape']
    div_rfield = bf['Field'][:].reshape(shape)

# smooth
div_smooth_ = dens_weight_smooth( div_rfield, dens_rfield, nmesh=1536, num_threads=64 ) 
div_smooth  = np.asarray( div_smooth_ )

# divergence VS. dark matter density
Ndens       = 24
densbins    = np.geomspace(0.75e-4,1.45e+4,Ndens+1,endpoint=True) 
dens_center = np.exp(0.5*(np.log(densbins)[1:]+np.log(densbins)[:-1]))
mean        = np.zeros(Ndens)
upstd       = np.zeros(Ndens)
bostd       = np.zeros(Ndens)
for i in range(Ndens):
    lbound    = densbins[i]
    ubound    = densbins[i+1]
    ratio_tem = div_smooth[(dens_rfield>=lbound)&(dens_rfield<ubound)]
    mean[i]   = np.mean( ratio_tem )
    upstd[i]  = np.std( ratio_tem[ratio_tem>mean[i]],ddof=1 )
    bostd[i]  = np.std( ratio_tem[ratio_tem<mean[i]],ddof=1 )

# Save
np.savez('divergence.npz', density=dens_center, div=mean, up_error=upstd, low_error=bostd)