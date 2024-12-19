#!/usr/bin/env python
# coding: utf-8

"""
Assign particles to mesh by using nbodykit
"""
import numpy as np
import bigfile
from mpi4py import MPI
from nbodykit.lab import FieldMesh
from nbodykit.source.catalog import HDFCatalog, ArrayCatalog
from nbodykit import CurrentMPIComm

comm = CurrentMPIComm.get()
rank = comm.rank

# Parameters used to specify the path
simulation = "/TNG"     
run        = "/TNG50-1"  
redshift   = "/z0"
matter     = "whim"     

# Paths
snap_path = "/..."+simulation+run+redshift+"/snapshots/snap_099.*"
grid_path = "/..."+simulation+run+redshift+"/dens_fields/"+matter

# Parameters
nmesh     = 1536
boxsize   = 35.0                 # Mpc/h
window    = 'pcs'
mp        = 1.672623e-24
xh        = 0.76
gamma     = 5.0/3.0
kb        = 1.38065e-16

# PartType0: gas
gas_cat     = HDFCatalog(snap_path, dataset='PartType0', comm=comm)
# Compute the gas temperature
mu          = 4. / ( 1.+3*xh+4*xh*(gas_cat['ElectronAbundance'].compute()) )
mu         *= mp
temperature = ( gamma-1 ) * ( (gas_cat['InternalEnergy'].compute())/kb ) * mu * 1.e10
# The masses of baryonic gas
mass_gas    = gas_cat['Masses'].compute()
# The coordinates of gas, and convert them from kpc/h to Mpc/h
coord_gas   = (gas_cat['Coordinates'].compute()) / 1.e3

# Choose gas cells
temp_mask       = (temperature > 1.e5)&(temperature < 1.e7)
mass_whim       = mass_gas[temp_mask]
coord_whim      = coord_gas[temp_mask,:] 
total_mass_rank = np.sum(mass_whim)
total_mass      = comm.allreduce( total_mass_rank, op=MPI.SUM )
mean_density    = total_mass / boxsize**3

# Delete useless data
gas_cat     = None 
temperature = None
mass_gas    = None
coord_gas   = None 

# Construct Catalog
catalog_dict = {'Coordinates':coord_whim,'Masses':mass_whim}
whim_cat     = ArrayCatalog(catalog_dict, comm=comm)
# Create mesh 
mesh         = whim_cat.to_mesh(Nmesh=nmesh,BoxSize=boxsize,dtype='f4',compensated=False, 
                                resampler=window, position='Coordinates',weight='Masses')
rfield       = mesh.compute(mode='real')
rfield      *= mean_density 

mesh_        = FieldMesh( rfield )

# Output
mesh_.save(grid_path+'/physical_dens_field_'+window+'.bigfile', mode='real', dataset='Field')


