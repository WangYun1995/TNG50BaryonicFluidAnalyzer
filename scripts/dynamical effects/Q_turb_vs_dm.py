import bigfile
import numpy as np
from dens_weight_smooth_para.dens_weight_smooth import dens_weight_smooth


# Parameters used to specify the path
simulation = "/TNG"      
run        = "/TNG50-1"  
redshift   = "/z0"
matter     = "gas"      

# Load the fields
dens_path  = '/...'+simulation+run+redshift+'/dens_fields/dm/dens_field_pcs.bigfile'
qturb_path = '/...'+simulation+run+redshift+'/qturb_fields/'+matter+'/qturb_field_pcs.bigfile'
with bigfile.File(dens_path) as bf:
    shape       = bf['Field'].attrs['ndarray.shape']
    dens_rfield = bf['Field'][:].reshape(shape)

with bigfile.File(qturb_path) as bf:
    shape      = bf['Field'].attrs['ndarray.shape']
    qturb_rfield = bf['Field'][:].reshape(shape)

# Smooth
qturb_smooth_ = dens_weight_smooth( qturb_rfield, dens_rfield, nmesh=1536, num_threads=64 ) 
qturb_smooth  = np.asarray( qturb_smooth_ )

#Turbulence trem VS dark matter density
Ndens       = 24
densbins    = np.geomspace(0.75e-4,1.45e+4,Ndens+1,endpoint=True) 
dens_center = np.exp(0.5*(np.log(densbins)[1:]+np.log(densbins)[:-1]))
mean        = np.zeros(Ndens)
upstd       = np.zeros(Ndens)
bostd       = np.zeros(Ndens)
for i in range(Ndens):
    lbound    = densbins[i]
    ubound    = densbins[i+1]
    ratio_tem = qturb_smooth[(dens_rfield>=lbound)&(dens_rfield<ubound)]
    mean[i]   = np.mean( ratio_tem )
    upstd[i]  = np.std( ratio_tem[ratio_tem>mean[i]],ddof=1 )
    bostd[i]  = np.std( ratio_tem[ratio_tem<mean[i]],ddof=1 )


# Save
np.savez('Q_turb.npz', density=dens_center, qturb=mean, up_error=upstd, low_error=bostd)
