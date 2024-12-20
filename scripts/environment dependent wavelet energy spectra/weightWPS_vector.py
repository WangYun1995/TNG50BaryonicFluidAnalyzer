import time
import numpy as np
from main import weightWPSs # see https://github.com/WangYun1995/WPSmesh/blob/main/WPSmesh/main.py 
from nbodykit.lab import BigFileMesh
from nbodykit import CurrentMPIComm

comm = CurrentMPIComm.get()
rank = comm.rank

# Parameters used to specify the path
simulation = "/TNG"            
run        = "/TNG50-1"  
redshift   = "/z0"
field      = "vel"
matter     = "gas"            

# Routine to print script status to command line, with elapsed time
def print_status(comm,start_time,message):
    if comm.rank == 0:
        elapsed_time = time.time() - start_time
        print('%d\ts: %s' % (elapsed_time,message))

# Load the mesh
path_mesh     = '/...'+simulation+run+redshift+"/"+field+'_fields/'+matter+'/'+field+'_turb_pcs_'
path_maskmesh = '/...'+simulation+run+redshift+'/dens_fields/dm/dens_field_pcs.bigfile'
path_wmesh    = '/...'+simulation+run+redshift+'/dens_fields/gas/dens_field_pcs.bigfile'
maskmesh      = BigFileMesh(path_maskmesh, 'Field', comm=comm)
wmesh         = BigFileMesh(path_wmesh, 'Field', comm=comm)

#--------------------------------------------------------
start_time = time.time()
print_status(comm,start_time,'Starting the calculations')
#--------------------------------------------------------

# Measure the WPSs from the mesh
Nscales   = 30
Ndens     = 3
bins_temp = np.geomspace(0.125,8.0,Ndens-1,endpoint=True) 
densbins  = np.pad( bins_temp, (1, 1), 'constant', constant_values=(0,1e+100) )

for i in range(3):
    mesh = BigFileMesh(path_mesh+str(i)+'.bigfile', 'Field', comm=comm)
    k_pseu, f_vol, env_WPS, global_WPS = weightWPSs( mesh, maskmesh, wmesh, Nscales, densbins, kmax=1.0, wavelet='iso_cwgdw',comm=comm)
    # Save results to the npz file
    if (rank==0):
      np.savez("WPSs_"+str(i)+".npz", k_pseu=k_pseu, f_vol=f_vol, env_WPS = 0.5*env_WPS, global_WPS = 0.5*global_WPS)

#--------------------------------------------------------
print_status(comm,start_time,'Done')
#--------------------------------------------------------
