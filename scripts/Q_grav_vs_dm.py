import bigfile
import numpy as np
from dens_weight_smooth_para.dens_weight_smooth import dens_weight_smooth

# Parameters used to specify the path
simulation = "/TNG"      
run        = "/TNG50-1"               
redshift   = "/z0"
matter     = "gas"       
zz         = 0.0
scale_fac  = 1./(1.+zz)
H0         = 67.74
Om         = 0.3089
factor     = -1.5*Om*(H0/scale_fac)**2

# Load the fields
dens_path = '/...'+simulation+run+redshift+'/dens_fields/dm/dens_field_pcs.bigfile'
qgrav_path = '/...'+simulation+run+redshift+'/qgrav_fields/'+matter+'/qgrav_field_pcs.bigfile'
with bigfile.File(dens_path) as bf:
    shape       = bf['Field'].attrs['ndarray.shape']
    dens_rfield = bf['Field'][:].reshape(shape)

with bigfile.File(qgrav_path) as bf:
    shape      = bf['Field'].attrs['ndarray.shape']
    qgrav_rfield = bf['Field'][:].reshape(shape)

# Smooth
qgrav_smooth = dens_weight_smooth( qgrav_rfield, dens_rfield, nmesh=1536, num_threads=64 ) 

# gravitational term VS dark matter density
Ndens       = 24
densbins    = np.geomspace(0.75e-4,1.45e+4,Ndens+1,endpoint=True) 
dens_center = np.exp(0.5*(np.log(densbins)[1:]+np.log(densbins)[:-1]))
mean        = np.zeros(Ndens)
upstd       = np.zeros(Ndens)
bostd       = np.zeros(Ndens)
for i in range(Ndens):
    lbound    = densbins[i]
    ubound    = densbins[i+1]
    ratio_tem = qgrav_smooth[(dens_rfield>=lbound)&(dens_rfield<ubound)]
    mean[i]   = np.mean( ratio_tem )
    upstd[i]  = np.std( ratio_tem[ratio_tem>mean[i]],ddof=1 )
    bostd[i]  = np.std( ratio_tem[ratio_tem<mean[i]],ddof=1 )

# Save
np.savez('Q_grav.npz', density=dens_center, qgrav=mean, up_error=upstd, low_error=bostd)