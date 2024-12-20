import numpy as np
from scipy import stats
from mpi4py import MPI
from nbodykit import CurrentMPIComm
from nbodykit.lab import BigFileMesh, FieldMesh


# Parameters used to specify the path
simulation = "/TNG"     
run        = "/TNG50-1"  
redshift   = "/z0"
matter     = "gas"       

# Bins of the dark matter density field
Ndens       = 24
densbins    = np.geomspace(0.75e-4,1.45e+4,Ndens+1,endpoint=True) 
dens_center = np.exp(0.5*(np.log(densbins)[1:]+np.log(densbins)[:-1]))

# Load the density field
dens_path   = '/...'+simulation+run+redshift+'/dens_fields/'+matter+'/dens_field_pcs.bigfile'
dm_path     = '/...'+simulation+run+redshift+'/dens_fields/dm/dens_field_pcs.bigfile'
dens_mesh   = BigFileMesh(dens_path,'Field')
dm_mesh     = BigFileMesh(dm_path,'Field')
dens_rfield = dens_mesh.compute(mode='real').copy()
dm_rfield   = dm_mesh.compute(mode='real').copy()

# Load the thermal energy per unit mass
therm_path   = '/...'+simulation+run+redshift+'/energy_fields/'+matter+'/energy_field_pcs.bigfile'
therm_mesh   = BigFileMesh(therm_path,'Field')
therm_rfield = therm_mesh.compute(mode='real').copy()

# Compute the energy
energy_rfield = dens_rfield * therm_rfield

# Compute the energy VS. dark mater density
mean  = np.zeros(Ndens)
upstd = np.zeros(Ndens)
bostd = np.zeros(Ndens)
for i in range(Ndens):
    lbound    = densbins[i]
    ubound    = densbins[i+1]
    ratio_tem = energy_rfield[(dm_rfield>=lbound)&(dm_rfield<ubound)]
    mean[i]   = np.mean( ratio_tem )
    upstd[i]  = np.std( ratio_tem[ratio_tem>mean[i]],ddof=1 )
    bostd[i]  = np.std( ratio_tem[ratio_tem<mean[i]],ddof=1 )

# Save
np.savez('therm_energy.npz', density=dens_center, energy=mean, up_error=upstd, low_error=bostd)