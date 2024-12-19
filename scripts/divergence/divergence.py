import numpy as np
from nbodykit import CurrentMPIComm
from nbodykit.lab import BigFileMesh, FieldMesh

comm = CurrentMPIComm.get()
rank = comm.rank


def VecField_divergence( mesh_tuple ):
  '''
  Compute the divergence of the 3D vector field, e.g. the velocity field.

  ---------------
  Author: iterrec
  https://github.com/mschmittfull/iterrec
  '''
  out_cfield = None
  for component in range(3):

    # copy so we don't modify the input
    input_cfield = mesh_tuple[component].compute(mode='complex').copy()
    def derivative(k, v, i=component):
      return 1j*k[i]*v
    
    if out_cfield is None:
      out_cfield = input_cfield.apply(derivative, kind='wavenumber')
    else:
      out_cfield += input_cfield.apply(derivative, kind='wavenumber')

    del input_cfield
  
  return FieldMesh( out_cfield.c2r() )
#----------------------------------------------------------------------------------------------


# Parameters used to specify the path
simulation = "/TNG"      
run        = "/TNG50-1"  
redshift   = "/z0"
matter     = "gas"     

# Load the velocity field
vel_mesh   = {}
path_mesh  = '/...'+simulation+run+redshift+'/vel_fields/'+matter+'/vel_turb_pcs_'
for i in range(3):
  vel_mesh[i] = BigFileMesh(path_mesh+str(i)+'.bigfile', 'Field', comm=comm)

# Compute the divergence
div_mesh = VecField_divergence( vel_mesh )

# Save
div_mesh.save('/...'+simulation+run+redshift+'/div_fields/'+matter+'/div_turb_pcs.bigfile')