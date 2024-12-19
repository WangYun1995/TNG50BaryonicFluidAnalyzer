import numpy as np
from scipy import stats
from mpi4py import MPI
from nbodykit.lab import BigFileMesh
from nbodykit import CurrentMPIComm

comm = CurrentMPIComm.get()
rank = comm.rank

# Parameters used to specify the path
simulation = "/TNG"      
run        = "/TNG50-1"  
redshift   = "/z0"

# Split the density into Ndens bins
Ndens      = 25
bins_temp  = np.geomspace(1e-4,1e+4,Ndens-1,endpoint=True) 
densbins   = np.pad( bins_temp, (1, 1), 'constant', constant_values=(0,1e+10) )

# Load the fields
baryon_rfield_tuple = {}
baryon_com        = ['hot','whim','warm','cold']
root_path         = '/...'+simulation+run+redshift+'/dens_fields/'
for i in range(len(baryon_com)):
  baryon_mesh            = BigFileMesh(root_path+baryon_com[i]+'/physical_dens_field_pcs.bigfile', 'Field', comm=comm)
  baryon_rfield_tuple[i] = baryon_mesh.compute(mode='real') 

# Load the dark matter density field
dm_path   = '/...'+simulation+run+redshift+'/dens_fields/dm/dens_field_pcs.bigfile'
dm_mesh   = BigFileMesh(dm_path, 'Field', comm=comm)
dm_rfield = dm_mesh.compute(mode='real')

# Compute the mass fraction
mass_frac_rank = np.empty( (len(baryon_com),len(densbins)-1) )
for i in range(len(baryon_com)):
    mass_frac_rank[i,:], _, _ = stats.binned_statistic( np.ravel(dm_rfield), np.ravel(baryon_rfield_tuple[i]), 'sum', bins=densbins)

# Total baryon
baryon_total    = sum( baryon_rfield_tuple[i] for i in range(len(baryon_com)) )
mass_total_rank = np.sum(baryon_total)

# Reduction
mass_frac  = comm.allreduce(mass_frac_rank, op=MPI.SUM)
mass_total = comm.allreduce(mass_total_rank, op=MPI.SUM)
mass_frac /= mass_total

# Save
if (rank==0):
    np.savez('cum_mass_frac_vs_dm.npz', density=densbins, mass_fraction=mass_frac)