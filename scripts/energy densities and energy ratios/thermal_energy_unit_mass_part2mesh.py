#!/usr/bin/env python
# coding: utf-8

"""
Assign Internal (thermal) energy per unit mass  to mesh by using nbodykit
"""

import time
import numpy as np
from nbodykit.lab import *
from nbodykit.source.catalog import HDFCatalog
from nbodykit import CurrentMPIComm

comm = CurrentMPIComm.get()
rank = comm.rank

#----------------------------------------------------------------------#
# Routine to print script status to command line, with elapsed time
def print_status(comm,start_time,message):
    if comm.rank == 0:
        elapsed_time = time.time() - start_time
        print('%d\ts: %s' % (elapsed_time,message))
#----------------------------------------------------------------------#

simulation = "/TNG"     
run        = "/TNG50-1"  
# matter component
matter    = "gas"
# redshift
redshift  = "/z0p5"


# Paths
snap_path = "/..."+simulation+run+redshift+"/snapshots/snap_099.*"
grid_path = "/..."+simulation+run+redshift+"/energy_fields/"+matter+"/"

# Parameters
nmesh     = 1536
boxsize   = 35.0                 # Mpc/h
window    = 'pcs'

# Set start time, and print first status message
start_time = time.time()
print_status(comm,start_time,'Starting conversion')

# PartType0: gas
in_cat                 = HDFCatalog(snap_path, dataset='PartType0', comm=comm)
# Convert particle positions from kpc/h to Mpc/h
in_cat['Coordinates'] /= 1e3
# Create meshs
density_mesh = in_cat.to_mesh(Nmesh=nmesh,BoxSize=boxsize,dtype='f4',compensated=False, 
                            resampler=window, position='Coordinates',weight='Masses')
wenergy_mesh = in_cat.to_mesh(Nmesh=nmesh,BoxSize=boxsize,dtype='f4',compensated=False, 
                            resampler=window, position='Coordinates',weight='Masses', value = 'InternalEnergy')
# Fields
density_field = density_mesh.compute(mode='real')
wenergy_field = wenergy_mesh.compute(mode='real')
with np.errstate(divide='ignore', invalid='ignore'):
    energy_field = wenergy_field/density_field
energy_field[density_field==0] = 0.0
energy_mesh = FieldMesh(energy_field)

print_status(comm,start_time,'Created mesh')

# Save the mesh to disk
print_status(comm,start_time,'Starting file output')

energy_mesh.save(grid_path+'energy_field_'+window+'.bigfile', mode='real', dataset='Field')

print_status(comm,start_time,'mesh data set is generated.')
