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
qtot_path = '/...'+simulation+run+redshift+'/qtot_fields/'+matter+'/qtot_field_pcs.bigfile'
# mesh
with bigfile.File(dens_path) as bf:
    shape       = bf['Field'].attrs['ndarray.shape']
    dens_rfield = bf['Field'][:].reshape(shape)

with bigfile.File(qtot_path) as bf:
    shape       = bf['Field'].attrs['ndarray.shape']
    qtot_rfield = bf['Field'][:].reshape(shape)

# Smooth
qtot_smooth_ = dens_weight_smooth( qtot_rfield, dens_rfield, nmesh=1536, num_threads=64 ) 
qtot_smooth  = np.asarray( qtot_smooth_ )

# Total dynamical term VS dark matter density
Ndens       = 24
densbins    = np.geomspace(0.75e-4,1.45e+4,Ndens+1,endpoint=True) 
dens_center = np.exp(0.5*(np.log(densbins)[1:]+np.log(densbins)[:-1]))
mean        = np.zeros(Ndens)
upstd       = np.zeros(Ndens)
bostd       = np.zeros(Ndens)
for i in range(Ndens):
    lbound    = densbins[i]
    ubound    = densbins[i+1]
    ratio_tem = qtot_smooth[(dens_rfield>=lbound)&(dens_rfield<ubound)]
    mean[i]   = np.mean( ratio_tem )
    upstd[i]  = np.std( ratio_tem[ratio_tem>mean[i]],ddof=1 )#/np.sqrt(np.size(ratio_tem[ratio_tem>mean[i]]))
    bostd[i]  = np.std( ratio_tem[ratio_tem<mean[i]],ddof=1 )#/np.sqrt(np.size(ratio_tem[ratio_tem>mean[i]]))


# Save
np.savez('Q_tot.npz', density=dens_center, qtot=mean, up_error=upstd, low_error=bostd)