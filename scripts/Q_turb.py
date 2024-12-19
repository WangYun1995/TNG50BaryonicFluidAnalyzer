import numpy as np
from nbodykit import CurrentMPIComm
from nbodykit.lab import BigFileMesh, FieldMesh

comm = CurrentMPIComm.get()
rank = comm.rank

def SymVelDeri( vi, vj, i, j, scale ):

  # copy so we don't modify the input
  vi_cfield = vi.compute(mode='complex').copy()
  vj_cfield = vj.compute(mode='complex').copy()
  def nth_partial_deri(n):
    def derivative(k, v):
      k2     = sum( k[i]**2 for i in range(3) )
      scale2 = scale**2
      kernel = np.exp(-0.5*scale2*k2)
      return 1j*k[n]*kernel*v
    return derivative
  
  vidj_cfield = vi_cfield.apply(nth_partial_deri(j), kind='wavenumber')
  vjdi_cfield = vj_cfield.apply(nth_partial_deri(i), kind='wavenumber')

  del vi_cfield
  del vj_cfield

  Sij = 0.5*( vidj_cfield.c2r()+vjdi_cfield.c2r() )
  
  return Sij
#----------------------------------------------------------------------------------------------

def SS( mesh_tuple, scale ):
  dia_indx     = np.array([[0,0],[1,1],[2,2]])
  nodia_indx   = np.array([[0,1],[0,2],[1,2]])
  dia_rfield   = None
  nodia_rfield = None 
  for i in range(3):
    ii    = dia_indx[i,0]
    jj    = dia_indx[i,1]
    sij   = SymVelDeri( mesh_tuple[ii], mesh_tuple[jj], ii, jj, scale )
    sij2  = sij**2
    if dia_rfield is None:
      dia_rfield = sij2
    else:
      dia_rfield += sij2
  for i in range(3):
    ii    = nodia_indx[i,0]
    jj    = nodia_indx[i,1]
    sij   = SymVelDeri( mesh_tuple[ii], mesh_tuple[jj], ii, jj, scale )
    sij2  = sij**2
    if nodia_rfield is None:
       nodia_rfield = sij2
    else:
      nodia_rfield += sij2
  ss_rfield = dia_rfield + 2*nodia_rfield
  return ss_rfield
#----------------------------------------------------------------------------------------------

def Curl2( mesh_tuple, scale ):
  '''
  Compute the curl (vorticity) of the 3D vector field, e.g. the velocity field.
  '''
  out_rfield    = None
  input_cfields = {}
  for component in range(3):
    # copy so we don't modify the input
    input_cfields[component] = mesh_tuple[component].compute(mode='complex').copy()
  
  for component in range(3):
    jj = (component+1)%3
    kk = (component+2)%3

    def derivative(i):
      def calculate(k, v):
        k2     = sum( k[i]**2 for i in range(3) )
        scale2 = scale**2
        kernel = np.exp(-0.5*scale2*k2)
        return 1j*k[i]*kernel*v
      return calculate 
    
    cfield_jj   = input_cfields[kk].apply( derivative(jj) )
    cfield_kk   = input_cfields[jj].apply( derivative(kk) )
    out_cfield  = cfield_jj-cfield_kk
    tem_rfield  = out_cfield.c2r()
    tem_rfield2 = tem_rfield**2
    if out_rfield is None:
      out_rfield  = tem_rfield2
    else:
      out_rfield += tem_rfield2
    
  del input_cfields

  return out_rfield
#----------------------------------------------------------------------------------------------

# Parameters used to specify the path
simulation = "/TNG"      
run        = "/TNG50-1"  
redshift   = "/z0"
matter     = "gas"       
zz         = 0.0
scale_fac  = 1/(1.+zz)

# Load the velocity field
vel_meshs = {}
vel_path  = '/media/hep-cosmo/data'+simulation+run+redshift+'/vel_fields/'+matter+'/vel_field_pcs_'
for i in range(3):
  vel_meshs[i] = BigFileMesh(vel_path+str(i)+'.bigfile', 'Field', comm=comm)

# Compute Qturb
w2 = Curl2( vel_meshs, 0.03 )
ss = SS( vel_meshs, 0.03 )
Qturb_rfield = (0.5*w2 - ss)/scale_fac
Qturb_mesh   = FieldMesh( Qturb_rfield )

# Save
Qturb_mesh.save('/media/hep-cosmo/data'+simulation+run+redshift+'/qturb_fields/'+matter+'/qturb_field_pcs.bigfile')



