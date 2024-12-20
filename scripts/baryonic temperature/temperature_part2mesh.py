#!/usr/bin/env python
# coding: utf-8

"""
Compute the mass fraction of different baryonic components by using nbodykit
"""

import time
import numpy as np
import h5py
import glob
from mpi4py import MPI
from nbodykit.lab import *
from nbodykit.source.catalog import HDFCatalog
from nbodykit import CurrentMPIComm

comm = CurrentMPIComm.get()
rank = comm.rank

simulation = "/TNG"    
run        = "/TNG50-1" 
# redshift
redshift  = "/z0"
snap_num  = "099"
# Parameters
nmesh     = 1536
boxsize   = 35.0                 # Mpc/h
window    = 'pcs'

# Paths
snap_path = "/..."+simulation+run+redshift+"/snapshots/snap_"+snap_num+".*"
grid_path = "/..."+simulation+run+redshift+"/temp_fields/gas/"

# Parameters
mp    = 1.672623e-24
xh    = 0.76
gamma = 5.0/3.0
kb    = 1.38065e-16

# PartType0: gas
gas_cat  = HDFCatalog(snap_path, dataset='PartType0', comm=comm)

# Compute the gas temperature
mu                        = 4. / ( 1.+3*xh+4*xh*(gas_cat['ElectronAbundance']) )
mu                       *= mp
gas_cat['InternalEnergy'] = ( gamma-1 ) * ( (gas_cat['InternalEnergy'])/kb ) * mu * 1.e10

# Convert particle positions from kpc/h to Mpc/h
gas_cat['Coordinates'] /= 1e3

# Create meshs
density_mesh = gas_cat.to_mesh(Nmesh=nmesh,BoxSize=boxsize,dtype='f4',compensated=False, 
                            resampler=window, position='Coordinates',weight='Masses')
wtemp_mesh   = gas_cat.to_mesh(Nmesh=nmesh,BoxSize=boxsize,dtype='f4',compensated=False, 
                            resampler=window, position='Coordinates',weight='Masses', value = 'InternalEnergy')
# Fields
density_field = density_mesh.compute(mode='real')
wtemp_field   = wtemp_mesh.compute(mode='real')
with np.errstate(divide='ignore', invalid='ignore'):
    temp_field = wtemp_field/density_field
temp_field[density_field==0] = 0.0
temp_mesh = FieldMesh(temp_field)

# Output
temp_mesh.save(grid_path+'temp_field_'+window+'.bigfile', mode='real', dataset='Field')
