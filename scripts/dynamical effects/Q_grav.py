import numpy as np
from nbodykit import CurrentMPIComm
from nbodykit.lab import BigFileMesh, FieldMesh

comm = CurrentMPIComm.get()
rank = comm.rank

# Parameters used to specify the path
simulation = "/TNG"      
run        = "/TNG50-1"  
redshift   = "/z0"
matter     = "gas"       
zz         = 0.0
scale_fac  = 1./(1.+zz)
H0         = 67.74
Om         = 0.3089

# Load mesh
dens_path  = '/...'+simulation+run+'/z0/dens_fields/tot/dens_field_pcs.bigfile'
dens_mesh  = BigFileMesh(dens_path, 'Field', comm=comm)
Lbox       = dens_mesh.attrs['BoxSize'][0] # Unit: Mpc/h
Nmesh      = dens_mesh.attrs['Nmesh'][0]   # Integer

# Compute the Q_grav
factor      = -1.5*Om*(H0/scale_fac)**2
dens_rfield = dens_mesh.compute(mode='real').copy()
Q_grav      = factor*( dens_rfield-1. )
Q_grav_mesh = FieldMesh( Q_grav )

# Save
Q_grav_mesh.save('/...'+simulation+run+redshift+'/qgrav_fields/'+matter+'/qgrav_field_pcs.bigfile')
