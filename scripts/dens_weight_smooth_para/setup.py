from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("dens_weight_smooth", ["dens_weight_smooth.pyx"],
              include_dirs=[np.get_include()],
              extra_compile_args=["-fopenmp"],
              extra_link_args=["-fopenmp"])
]

setup(
    name="dens_weight_smooth",
    ext_modules=cythonize(extensions, annotate=True),
    
    )

