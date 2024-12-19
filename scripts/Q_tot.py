import numpy as np
from nbodykit import CurrentMPIComm
from nbodykit.lab import BigFileMesh, FieldMesh

comm = CurrentMPIComm.get()
rank = comm.rank

# Parameters used to specify the path
simulation = "/TNG"      # EAGLE, Illustris, SIMBA, TNG
run        = "/TNG50-1"  # hydro: RefL0100N1504, Illustris-1, m100n1024, TNG100-1, TNG300-1
                         # DMO: DMONLYL0100N1504, Illustris-1-Dark, m100n1024-DMO, TNG100-1-Dark, TNG300-1-Dark
redshift   = "/z0"
matter     = "gas"       # hydro: tot, dm, ba; DMO: dmo
turb_path  = '/...'+simulation+run+redshift+'/qturb_fields/'+matter+'/qturb_field_pcs.bigfile'
th_path    = '/...'+simulation+run+redshift+'/qth_fields/'+matter+'/qth_field_pcs.bigfile'
grav_path  = '/...'+simulation+run+redshift+'/qgrav_fields/'+matter+'/qgrav_field_pcs.bigfile'
exp_path   = '/...'+simulation+run+redshift+'/qexp_fields/'+matter+'/qexp_field_pcs.bigfile'

# Load mesh
turb_mesh   = BigFileMesh(turb_path, 'Field', comm=comm)
th_mesh     = BigFileMesh(th_path, 'Field', comm=comm)
grav_mesh   = BigFileMesh(grav_path, 'Field', comm=comm)
exp_mesh    = BigFileMesh(exp_path, 'Field', comm=comm)

# Compute Q_tot
turb_rfield = turb_mesh.compute(mode='real').copy()
th_rfield   = th_mesh.compute(mode='real').copy()
grav_rfield = grav_mesh.compute(mode='real').copy()
exp_rfield  = exp_mesh.compute(mode='real').copy()

qtot_rfield = turb_rfield + th_rfield + grav_rfield + exp_rfield
qtot_mesh   = FieldMesh( qtot_rfield )

# Save
qtot_mesh.save('/...'+simulation+run+redshift+'/qtot_fields/'+matter+'/qtot_field_pcs.bigfile')