#!/usr/bin/env python
# coding: utf-8

"""
Assign particles to mesh by using nbodykit
"""
import numpy as np
import bigfile
import h5py
import glob
from mpi4py import MPI
from nbodykit.lab import FieldMesh, MultipleSpeciesCatalog
from nbodykit.source.catalog import HDFCatalog, ArrayCatalog
from nbodykit import CurrentMPIComm

comm = CurrentMPIComm.get()
rank = comm.rank

#-----------------------------------------------#
def effective_path_PartType5( snapshots_path, comm=comm ):
    # Exclude files that do not contain black holes
    rank = comm.rank
    if (rank==0):
        all_files   = glob.glob(snapshots_path)
        valid_path = []
        for hdf5 in all_files:
            with h5py.File(hdf5,'r') as f:
                if 'PartType5' in f:
                    valid_path.append(hdf5)
    else:
        valid_path = None

    valid_path = comm.bcast(valid_path, root=0)
    return valid_path
#-----------------------------------------------#

# Parameters used to specify the path
simulation = "/TNG"     
run        = "/TNG50-1" 
redshift   = "/z0"
matter     = "cold"       

# Paths
snap_path = "/..."+simulation+run+redshift+"/snapshots/snap_021.*"
grid_path = "/..."+simulation+run+redshift+"/dens_fields/"+matter

# Parameters
nmesh     = 1536
boxsize   = 35.0                 # Mpc/h
window    = 'pcs'
mp        = 1.672623e-24
xh        = 0.76
gamma     = 5.0/3.0
kb        = 1.38065e-16

# PartType4: stars, PartType5: black holes
bhs_path     = effective_path_PartType5( snap_path, comm=comm )
star_cat     = HDFCatalog(snap_path, dataset='PartType4', comm=comm)
bhs_cat      = HDFCatalog(bhs_path, dataset='PartType5', comm=comm)

# The masses of stars and black holes
mass_star       = star_cat['Masses'].compute()
mass_bhs        = bhs_cat['Masses'].compute()
# The mean physical density
total_mass_rank = np.sum(mass_star) + np.sum(mass_bhs)
total_mass      = comm.allreduce( total_mass_rank, op=MPI.SUM )
mean_density    = total_mass / boxsize**3

# The coordinates of gas, and convert them from kpc/h to Mpc/h
star_cat['Coordinates'] /= 1.e3
bhs_cat['Coordinates']  /= 1.e3

# Combine Catalog
combine_cat = MultipleSpeciesCatalog( ['stars','BHs'], star_cat, bhs_cat)
# Create mesh 
mesh         = combine_cat.to_mesh(Nmesh=nmesh,BoxSize=boxsize,dtype='f4',compensated=False, 
                                resampler=window, position='Coordinates',weight='Masses')
rfield       = mesh.compute(mode='real')
rfield      *= mean_density 

mesh_        = FieldMesh( rfield )

# Output
mesh_.save(grid_path+'/physical_dens_field_'+window+'.bigfile', mode='real', dataset='Field')


