import numpy as np
import bigfile


# Parameters used to specify the path
simulation = "/TNG"      
run        = "/TNG50-1"                      
redshift   = "/z0"
field_typ  = "/dens"
path_hot   = '/...'+simulation+run+redshift+field_typ+'_fields/hot/physical_dens_field_pcs.bigfile'
path_whim  = '/...'+simulation+run+redshift+field_typ+'_fields/whim/physical_dens_field_pcs.bigfile'
path_warm  = '/...'+simulation+run+redshift+field_typ+'_fields/warm/physical_dens_field_pcs.bigfile'
path_cold  = '/...'+simulation+run+redshift+field_typ+'_fields/cold/physical_dens_field_pcs.bigfile'
path_dm    = '/...'+simulation+run+redshift+field_typ+'_fields/dm/dens_field_pcs.bigfile'

# Load dark matter
dm_dens_mean = 7.224625837327304   # The mean dark matter density of the TNG50-1 simulation, i.e. \bar\rho_{dm}
with bigfile.File(path_dm) as bf:
    shape    = bf['Field'].attrs['ndarray.shape']
    dm_dens  = bf['Field'][:].reshape(shape)

# Load baryons
dens_list  = []
paths      = [path_hot, path_whim, path_warm, path_cold]
for i, path in enumerate(paths):
    with bigfile.File(path) as bf:
        shape = bf['Field'].attrs['ndarray.shape']
        dens  = bf['Field'][:].reshape(shape)
    dens_list.append( dens )

# total matter
baryon_dens = sum( dens_list[i] for i in range(len(paths)) )
total_dens  = dm_dens_mean*dm_dens + baryon_dens

# Compute baryon fraction
with np.errstate(divide='ignore', invalid='ignore'):
    baryon_frac = baryon_dens / total_dens
baryon_frac[total_dens==0] = 0.0
baryon_frac = np.ravel( baryon_frac )[::1728] 

dm_dens_ = np.ravel( dm_dens )[::1728]

# Save
np.savez('baryon_vs_dm_z0.npz', baryon_fraction = baryon_frac, 
                             dm = dm_dens_)
