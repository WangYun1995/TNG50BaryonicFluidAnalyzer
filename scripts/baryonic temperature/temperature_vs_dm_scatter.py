import numpy as np
from scipy import stats
from mpi4py import MPI
from nbodykit import CurrentMPIComm
from nbodykit.lab import BigFileMesh, FieldMesh


# Parameters used to specify the path
simulation = "/TNG"      # EAGLE, Illustris, SIMBA, TNG
run        = "/TNG50-1"  # hydro: RefL0100N1504, Illustris-1, m100n1024, TNG100-1, TNG300-1
                         # DMO: DMONLYL0100N1504, Illustris-1-Dark, m100n1024-DMO, TNG100-1-Dark, TNG300-1-Dark
redshift   = "/z0"
matter     = "gas"       # hydro: tot, dm, ba; DMO: dmo

# Load the density field
dm_path     = '/media/hep-cosmo/data'+simulation+run+redshift+'/dens_fields/dm/dens_field_pcs.bigfile'
dm_mesh     = BigFileMesh(dm_path,'Field')
dm_rfield   = dm_mesh.compute(mode='real').copy()

# Load the temperature field
temp_path   = '/media/hep-cosmo/data'+simulation+run+redshift+'/temp_fields/'+matter+'/temp_field_pcs.bigfile'
temp_mesh   = BigFileMesh(temp_path,'Field')
temp_rfield = temp_mesh.compute(mode='real').copy()

# choose sub sets
temp_sub = np.ravel( temp_rfield )[::1728]
dm_sub   = np.ravel( dm_rfield )[::1728]

# Save
np.savez('temperature_vs_dm_scatter_z0.npz', temperature=temp_sub, dm=dm_sub)