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
redshift  = "/z0" # /z0p5, /z1, /z1p5, /z2, /z4
snap_num  = "099" # 067, 050, 040, 033, 021

# Paths
snap_path = "/..."+simulation+run+redshift+"/snapshots/snap_"+snap_num+".*"
save_path = "/..."+simulation+run+redshift+"/mass_fractions/"

# Parameters
mp    = 1.672623e-24
xh    = 0.76
gamma = 5.0/3.0
kb    = 1.38065e-16

# PartType0: gas
gas_cat  = HDFCatalog(snap_path, dataset='PartType0', comm=comm)
# PartType4: stars
star_cat = HDFCatalog(snap_path, dataset='PartType4', comm=comm)

# PartType5: black holes
# Exclude files that do not contain black holes
if (rank==0):
    all_hdf5   = glob.glob(snap_path)
    valid_hdf5 = []
    for hdf5 in all_hdf5:
        with h5py.File(hdf5,'r') as f:
            if 'PartType5' in f:
                valid_hdf5.append(hdf5)
else:
    valid_hdf5 = None

valid_hdf5 = comm.bcast(valid_hdf5, root=0)
bhs_cat  = HDFCatalog(valid_hdf5, dataset='PartType5', comm=comm)

# Compute the gas temperature
mu          = 4. / ( 1.+3*xh+4*xh*(gas_cat['ElectronAbundance'].compute()) )
mu         *= mp
temperature = ( gamma-1 ) * ( (gas_cat['InternalEnergy'].compute())/kb ) * mu * 1.e10

# The mass of baryonic matter
mass_gas     = gas_cat['Masses'].compute()
mass_star    = star_cat['Masses'].compute()
mass_bhs     = bhs_cat['Masses'].compute()

m_tot_rank   = np.sum( mass_gas ) + np.sum( mass_star ) + np.sum( mass_bhs )
m_hg_rank    = np.sum( mass_gas[temperature>1.e7] )
m_whim_rank  = np.sum( mass_gas[(temperature<1.e7)&(temperature>1.e5)] )
m_wg_rank    = np.sum( mass_gas[temperature<1.e5] )
m_cold_rank  = np.sum( mass_star ) + np.sum( mass_bhs )

# reduction
m_tot  = comm.allreduce(m_tot_rank, op=MPI.SUM)
m_hg   = comm.allreduce(m_hg_rank, op=MPI.SUM) 
m_whim = comm.allreduce(m_whim_rank, op=MPI.SUM) 
m_wg   = comm.allreduce(m_wg_rank, op=MPI.SUM) 
m_cold = comm.allreduce(m_cold_rank, op=MPI.SUM) 

if (rank==0):
    m_hg   /= m_tot
    m_whim /= m_tot
    m_wg   /= m_tot
    m_cold /= m_tot
    np.savez(save_path+'baryon_mass_frac.npz', hot=m_hg, whim=m_whim, warm=m_wg, cold=m_cold)

