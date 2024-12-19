import numpy as np
from nbodykit import CurrentMPIComm
from nbodykit.lab import BigFileMesh, FieldMesh

comm = CurrentMPIComm.get()
rank = comm.rank

def gauss_smooth( mesh, scale ):
  cfield           = mesh.compute(mode='complex').copy()
  def gauss(k,v):
    k2 = sum( k[i]**2 for i in range(3) )
    scale2 = scale**2
    kernel = np.exp(-0.5*scale2*k2)
    return v * kernel
  out_cfield = cfield.apply( gauss )
  out_rfield = out_cfield.c2r()
  return out_rfield
#-----------------------------------------------------------

def gradient( mesh, scale ):
  
  out_rfield_tuple = {}
  cfield           = mesh.compute(mode='complex').copy()
  for i in range(3):
    def derivative(axis):
      def calculate(k, v):
        k2     = sum( k[i]**2 for i in range(3) )
        scale2 = scale**2
        kernel = np.exp(-0.5*scale2*k2)
        return v * (1j * k[axis]) *kernel
      return calculate
    out_cfield          = cfield.apply( derivative(i) )
    out_rfield_tuple[i] = out_cfield.c2r()
  return out_rfield_tuple
#-----------------------------------------------------------

def laplace( mesh, scale ):

  cfield = mesh.compute(mode='complex').copy()
  def calcul_lap(k,v):
    k2     = sum( k[i]**2 for i in range(3) )
    scale2 = scale**2
    kernel = np.exp(-0.5*scale2*k2)
    return -k2*v*kernel
  out_cfield = cfield.apply( calcul_lap )
  out_rfield = out_cfield.c2r()
  return out_rfield
#-----------------------------------------------------------

# Parameters used to specify the path
simulation = "/TNG"     
run        = "/TNG50-1"  
redshift   = "/z0"
matter     = "gas"     
zz         = 0.0
scale_fac  = 1/(1.+zz)
dens_path  = '/...'+simulation+run+redshift+'/dens_fields/gas/dens_field_pcs.bigfile'
press_path = '/...'+simulation+run+redshift+'/press_fields/gas/press_field_pcs.bigfile'

# Load mesh
dens_mesh  = BigFileMesh(dens_path, 'Field', comm=comm)
press_mesh = BigFileMesh(press_path, 'Field', comm=comm)
Lbox       = dens_mesh.attrs['BoxSize'][0] # Unit: Mpc/h
Nmesh      = dens_mesh.attrs['Nmesh'][0]   # Integer

# Compute the Qth
dens_grad_rfields  = gradient( dens_mesh, 0.03 )
press_grad_rfields = gradient( press_mesh, 0.03 )
qth_a              = sum( dens_grad_rfields[i]*press_grad_rfields[i] for i in range(3) )
dens_smooth        = gauss_smooth( dens_mesh, 0.03 )
qth_a             /= dens_smooth
qth_b    = laplace( press_mesh, 0.03 )
qth      = (qth_a - qth_b)/dens_smooth
qth     /= scale_fac
qth_mesh = FieldMesh( qth )

# save
qth_mesh.save('/...'+simulation+run+redshift+'/qth_fields/'+matter+'/qth_field_pcs.bigfile')
